"""PA-to-Nebula event normalization copied from the experimental pipeline.
Only the methods needed to produce model input are included.
"""
import os
import re
import logging
from collections import defaultdict
from typing import Any, Dict, List
import numpy as np
import pandas as pd

class PAToNebulaPreprocessor:
    def __init__(self, output_dir: str = "nebula_format"):
        """
        Preprocessor class that converts PA data into Nebula format.

        Args:
            output_dir: Directory where outputs will be written.
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Nebula normalization patterns
        self.normalization_patterns = {
            # IP address normalization
            'private_ip': re.compile(r'\b(?:10\.\d+\.\d+\.\d+|172\.(?:1[6-9]|2\d|3[01])\.\d+\.\d+|192\.168\.\d+\.\d+)\b'),
            'public_ip': re.compile(r'\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b'),
            'loopback_ip': re.compile(r'\b127\.0\.0\.1\b'),
            'ipv6': re.compile(r'\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b'),

            # Hash normalization
            'sha256': re.compile(r'\b[a-fA-F0-9]{64}\b'),
            'sha1': re.compile(r'\b[a-fA-F0-9]{40}\b'),
            'md5': re.compile(r'\b[a-fA-F0-9]{32}\b'),

            # Domain normalization
            'domain': re.compile(r'\b[a-zA-Z0-9-]+\.(?:com|net|org|edu|gov|mil|int|arpa|info|biz|name|museum|coop|aero|[a-z]{2})\b'),

            # Linux path normalization
            'home_path': re.compile(r'/home/[^/\s]+'),
            'tmp_path': re.compile(r'/tmp/[^/\s]+'),
            'var_path': re.compile(r'/var/[^/\s]+'),
            'proc_path': re.compile(r'/proc/[^/\s]+'),
            'sys_path': re.compile(r'/sys/[^/\s]+'),

            # High (ephemeral) ports (exclude well-known ports)
            'high_port': re.compile(
                r'\b(?:102[4-9]|10[3-9]\d|1[1-9]\d{2}|[2-9]\d{3}|[1-5]\d{4}|6[0-4]\d{3}|65[0-4]\d{2}|655[0-2]\d|6553[0-5])\b'
            )
        }

        # Event statistics
        self.event_stats = defaultdict(int)

        logging.info("Initialized PA→Nebula preprocessor")

    def _is_missing_or_nan(self, value: Any) -> bool:
        """
        Safely detect missing values from pandas/NumPy (NaN, None, empty arrays/series, etc.).
        """
        if value is None:
            return True
        # NumPy arrays/Series: True if all elements are missing or container is empty
        try:
            if isinstance(value, (np.ndarray, pd.Series)):
                if value.size == 0:
                    return True
                try:
                    return bool(np.all(pd.isna(value)))
                except Exception:
                    return False
        except Exception:
            pass
        # Scalar or other types
        try:
            res = pd.isna(value)
            if isinstance(res, (np.ndarray, pd.Series, list, tuple)):
                try:
                    return bool(np.all(res))
                except Exception:
                    return False
            return bool(res)
        except Exception:
            return False

    def _ensure_list(self, value: Any) -> List[Any]:
        """
        Safely coerce event containers to a list:
        - dict → [dict]
        - list/tuple → list
        - NaN/None/invalid types → []
        - strings are not treated as a container here → []
        """
        if self._is_missing_or_nan(value):
            return []
        if isinstance(value, list):
            return value
        if isinstance(value, tuple):
            return list(value)
        if isinstance(value, dict):
            return [value]
        # Strings, numbers, and other iterables are not treated as event containers
        return []

    def _as_true(self, value: Any) -> bool:
        """
        Safely evaluate truthiness (supports arrays/series).
        """
        if self._is_missing_or_nan(value):
            return False
        try:
            if isinstance(value, (np.ndarray, pd.Series)):
                return bool(np.any(value))
            return bool(value)
        except Exception:
            return False

    def normalize_text(self, text: str) -> str:
        """
        Apply Nebula-style text normalization.

        Args:
            text: Text to normalize.

        Returns:
            str: Normalized text.
        """
        if not text:
            return text

        # Hashes
        text = self.normalization_patterns['sha256'].sub('<sha256>', text)
        text = self.normalization_patterns['sha1'].sub('<sha1>', text)
        text = self.normalization_patterns['md5'].sub('<md5>', text)

        # IP addresses
        text = self.normalization_patterns['loopback_ip'].sub('<loopback>', text)
        text = self.normalization_patterns['private_ip'].sub('<prvip>', text)
        text = self.normalization_patterns['public_ip'].sub('<pubip>', text)
        text = self.normalization_patterns['ipv6'].sub('<ipv6>', text)

        # Domains
        text = self.normalization_patterns['domain'].sub('<domain>', text)

        # Linux paths
        text = self.normalization_patterns['home_path'].sub('<home>', text)
        text = self.normalization_patterns['tmp_path'].sub('<tmp>', text)
        text = self.normalization_patterns['var_path'].sub('<var>', text)
        text = self.normalization_patterns['proc_path'].sub('<proc>', text)
        text = self.normalization_patterns['sys_path'].sub('<sys>', text)

        # High ports
        text = self.normalization_patterns['high_port'].sub('<highport>', text)

        return text

    def extract_file_events(self, pa_data: Dict[str, Any]) -> List[str]:
        """
        Convert PA 'Files' events into Nebula 'file_access' lines.

        Args:
            pa_data: PA JSON object.

        Returns:
            List[str]: Converted file event lines.
        """
        events = []

        files_data = self._ensure_list(pa_data.get('Files', []))
        if not files_data:
            return events

        for file_event in files_data:
            try:
                path = file_event.get('Path', '')
                # Handle NaN/empty/arrays
                if self._is_missing_or_nan(path) or not str(path).strip():
                    continue

                # Support multiple operation fields
                operations = []

                # BigQuery style: Read, Write, Delete boolean fields (array/NaN safe)
                if self._as_true(file_event.get('Read', False)):
                    operations.append('READ')
                if self._as_true(file_event.get('Write', False)):
                    operations.append('WRITE')
                if self._as_true(file_event.get('Delete', False)):
                    operations.append('DELETE')

                # Legacy: Execute field or textual hint
                if self._as_true(file_event.get('Execute', False)) or 'exec' in str(file_event).lower():
                    operations.append('EXECUTE')

                # Default operation if none specified
                if not operations:
                    operations = ['READ']

                # Normalize path
                normalized_path = self.normalize_text(str(path))

                # Emit one line per operation
                for operation in operations:
                    event_text = f"file_access {normalized_path} {operation}"
                    events.append(event_text)
                    self.event_stats['file_access'] += 1

            except Exception as e:
                logging.warning(f"Failed to process file event: {e}")
                continue

        return events

    def extract_network_events(self, pa_data: Dict[str, Any]) -> List[str]:
        """
        Convert PA 'Sockets' events into Nebula 'network_connect' lines.
        Paper-style example: network_connect <ip_or_host>:443

        Args:
            pa_data: PA JSON object.

        Returns:
            List[str]: Converted network event lines.
        """
        events = []

        sockets_data = self._ensure_list(pa_data.get('Sockets', []))
        if not sockets_data:
            return events

        for socket_event in sockets_data:
            try:
                addresses = []
                port = socket_event.get('Port', '')

                # Various address forms
                if 'Address' in socket_event:
                    address = socket_event.get('Address', '')
                    if address:
                        addresses.append(address)

                # BigQuery style: 'Hostnames' array
                if 'Hostnames' in socket_event:
                    hostnames = socket_event.get('Hostnames', [])
                    if hasattr(hostnames, '__iter__') and not isinstance(hostnames, str):
                        for hostname in hostnames:
                            if not self._is_missing_or_nan(hostname) and str(hostname).strip():
                                addresses.append(str(hostname))
                    elif not self._is_missing_or_nan(hostnames) and str(hostnames).strip():  # single hostname case
                        addresses.append(str(hostnames))

                # Skip if no addresses
                if not addresses:
                    continue

                # Emit one line per address
                for address in addresses:
                    normalized_address = self.normalize_text(str(address))

                    port_str = '' if self._is_missing_or_nan(port) else str(port)
                    if port_str.strip():
                        normalized_port = self.normalize_text(port_str)
                        event_text = f"network_connect {normalized_address}:{normalized_port}"
                    else:
                        event_text = f"network_connect {normalized_address}"

                    events.append(event_text)
                    self.event_stats['network_traffic'] += 1

            except Exception as e:
                logging.warning(f"Failed to process network event: {e}")
                continue

        return events

    def extract_dns_events(self, pa_data: Dict[str, Any]) -> List[str]:
        """
        Convert PA 'DNS' events into Nebula 'dns_query' lines.
        Paper-style example: dns_query <domain> A

        Args:
            pa_data: PA JSON object.

        Returns:
            List[str]: Converted DNS event lines.
        """
        events = []

        dns_data = self._ensure_list(pa_data.get('DNS', []))
        if not dns_data:
            return events

        for dns_event in dns_data:
            try:
                domain = ''
                record_type = 'A'  # default to A record (per paper-style example)

                # Support multiple DNS shapes
                if isinstance(dns_event, dict):
                    # Legacy: Domain or Query fields (NumPy-safe)
                    domain = dns_event.get('Domain', '')
                    if not domain:
                        domain = dns_event.get('Query', '')

                    # Record type if present
                    if 'Type' in dns_event:
                        record_type = str(dns_event.get('Type', 'A'))
                    elif 'RecordType' in dns_event:
                        record_type = str(dns_event.get('RecordType', 'A'))

                    # BigQuery: 'Queries' array with 'Hostname'
                    if not domain and 'Queries' in dns_event:
                        queries = dns_event.get('Queries', [])
                        if hasattr(queries, '__iter__') and not isinstance(queries, str):
                            for query in queries:
                                if isinstance(query, dict):
                                    if 'Hostname' in query:
                                        hostname = query.get('Hostname', '')
                                        if not self._is_missing_or_nan(hostname) and str(hostname).strip():
                                            domain = hostname
                                    if 'Type' in query:
                                        record_type = str(query.get('Type', 'A'))
                                    break
                elif isinstance(dns_event, str):
                    # Direct string domain
                    domain = dns_event

                if self._is_missing_or_nan(domain) or not str(domain).strip():
                    continue

                normalized_domain = self.normalize_text(str(domain))
                event_text = f"dns_query {normalized_domain} {record_type}"
                events.append(event_text)
                self.event_stats['dns_query'] += 1

            except Exception as e:
                logging.warning(f"Failed to process DNS event: {e}")
                continue

        return events

    def extract_command_events(self, pa_data: Dict[str, Any]) -> List[str]:
        """
        Convert PA 'Commands' events into Nebula 'apis execute_cmd ...' lines.

        Args:
            pa_data: PA JSON object.

        Returns:
            List[str]: Converted command event lines.
        """
        events = []

        commands_data = self._ensure_list(pa_data.get('Commands', []))
        if not commands_data:
            return events

        # Process each command event
        try:
            for cmd_event in commands_data:
                try:
                    if not isinstance(cmd_event, dict):
                        continue

                    # Safe emptiness check for various shapes (NumPy arrays, lists, strings)
                    def is_empty_or_none(value):
                        if value is None:
                            return True
                        try:
                            if hasattr(value, 'shape'):  # NumPy array
                                return value.size == 0
                            if hasattr(value, '__len__') and not isinstance(value, str):
                                return len(value) == 0
                            if isinstance(value, str):
                                return not value.strip()
                            return False
                        except:
                            return True

                    # Extract command from multiple possible fields (NumPy-safe)
                    command = cmd_event.get('Command')
                    if is_empty_or_none(command):
                        command = cmd_event.get('Cmd', '')

                    # Extract args from multiple possible fields
                    args = cmd_event.get('Args')
                    if is_empty_or_none(args):
                        args = cmd_event.get('Arguments', [])

                    if is_empty_or_none(command):
                        continue

                    # Convert arrays/lists to string safely
                    def safe_convert_to_string(value):
                        if value is None:
                            return ''
                        try:
                            if hasattr(value, 'tolist'):  # NumPy array
                                value = value.tolist()
                            if hasattr(value, '__iter__') and not isinstance(value, str):
                                return ' '.join(str(item) for item in value if item is not None)
                            else:
                                return str(value)
                        except:
                            return str(value) if value is not None else ''

                    command_str = safe_convert_to_string(command)

                    if is_empty_or_none(command_str):
                        continue

                    normalized_command = self.normalize_text(command_str)

                    # Base event text
                    event_text = f"apis execute_cmd {normalized_command}"

                    # Append arguments if present
                    if not is_empty_or_none(args):
                        try:
                            args_str = safe_convert_to_string(args)
                            if not is_empty_or_none(args_str):
                                normalized_args = self.normalize_text(args_str)
                                event_text = f"apis execute_cmd {normalized_command} {normalized_args}"
                        except:
                            # Swallow arg conversion errors; keep the command
                            pass

                    events.append(event_text)
                    self.event_stats['api_call'] += 1

                except Exception as e:
                    logging.warning(f"Failed to process command event: {e}")
                    continue
        except Exception as e:
            logging.warning(f"Failed to process Commands data: {e}")

        return events

    def extract_package_info_events(self, pa_data: Dict[str, Any]) -> List[str]:
        """
        Convert top-level Package info into Nebula-style context lines.

        Args:
            pa_data: PA JSON object.

        Returns:
            List[str]: Converted package info lines.
        """
        events = []

        try:
            # In BigQuery exports, package info may appear at the top level
            package_info = pa_data.get('Package', {})

            # Package.Name (NumPy-safe)
            package_name = package_info.get('name') or package_info.get('Name') or pa_data.get('package.Name', '')
            if not self._is_missing_or_nan(package_name):
                name_str = str(package_name)
                if name_str.strip():
                    # Normalize name (lowercase, special chars normalized)
                    normalized_name = self.normalize_text(name_str.lower())
                    event_text = f"package_info name {normalized_name}"
                    events.append(event_text)
                    self.event_stats['package_name'] += 1

            # Package.Version (NumPy-safe)
            package_version = package_info.get('version') or package_info.get('Version') or pa_data.get('package.Version', '')
            if not self._is_missing_or_nan(package_version):
                version_str = str(package_version)
                if version_str.strip():
                    normalized_version = self.normalize_text(version_str)
                    event_text = f"package_info version {normalized_version}"
                    events.append(event_text)
                    self.event_stats['package_version'] += 1

            # Package.Ecosystem if present (NumPy-safe)
            package_ecosystem = package_info.get('ecosystem') or package_info.get('Ecosystem') or pa_data.get('package.Ecosystem', '')
            if not self._is_missing_or_nan(package_ecosystem):
                eco_str = str(package_ecosystem)
                if eco_str.strip():
                    normalized_ecosystem = self.normalize_text(eco_str.lower())
                    event_text = f"package_info ecosystem {normalized_ecosystem}"
                    events.append(event_text)
                    self.event_stats['package_ecosystem'] += 1

        except Exception as e:
            logging.warning(f"Failed to process package info event: {e}")

        return events

    def convert_pa_to_nebula_format(self, pa_data: Dict[str, Any]) -> str:
        """
        Convert a single PA JSON object into a Nebula-style text line.

        Args:
            pa_data: PA JSON object (BigQuery shape supported).

        Returns:
            str: Nebula-style concatenated text.
        """
        all_events = []

        # Package info (context) first
        package_events = self.extract_package_info_events(pa_data)

        # BigQuery structure: Analysis > stages > events
        analysis_data = pa_data.get('Analysis', {})
        if self._is_missing_or_nan(analysis_data) or not isinstance(analysis_data, dict):
            analysis_data = {}

        # Merge events across stages (install, import, execute)
        combined_stage_data = {}
        for stage in ['install', 'import', 'execute']:
            stage_data = analysis_data.get(stage, {})
            if not isinstance(stage_data, dict) or self._is_missing_or_nan(stage_data):
                continue
            # Collect events per stage
            for event_type in ['Files', 'Sockets', 'DNS', 'Commands']:
                if event_type in stage_data:
                    if event_type not in combined_stage_data:
                        combined_stage_data[event_type] = []
                    stage_events_raw = stage_data.get(event_type, [])
                    stage_events = self._ensure_list(stage_events_raw)
                    if stage_events:
                        combined_stage_data[event_type].extend(stage_events)

        # Prefer merged BigQuery-style data; fall back to legacy top-level fields
        effective_data = combined_stage_data if combined_stage_data else pa_data

        # Extract four core event families
        file_events = self.extract_file_events(effective_data)
        network_events = self.extract_network_events(effective_data)
        dns_events = self.extract_dns_events(effective_data)
        command_events = self.extract_command_events(effective_data)

        # Assemble: package info first for context, then events
        all_events.extend(package_events)
        all_events.extend(file_events)
        all_events.extend(network_events)
        all_events.extend(dns_events)
        all_events.extend(command_events)

        # Join into a single line (space-separated)
        nebula_text = ' '.join(all_events)

        return nebula_text
