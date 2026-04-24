"""LADAR preprocessing: LLM-based APR for dynamic-analysis failure logs.

Public components
-----------------
- install_apr.RefinedErrorMessageProcessor: classifies install-phase stderr
  into 9 categories (Peng et al.) aligned with Table 6 of the paper.
- install_apr.TargetedPromptGenerator: single label-agnostic prompt for
  install-phase repair.
- execute_apr.ExecuteStageErrorMessageProcessor: classifies execute-phase
  stderr using Python's built-in exception hierarchy.
- execute_apr.Stage1PromptGenerator / Stage2PromptGenerator /
  CodeContextExtractor / Postprocessor: the two-stage execute-phase
  repair pipeline described in Section 4 of the paper.
"""
from . import install_apr, execute_apr

__all__ = ["install_apr", "execute_apr"]
