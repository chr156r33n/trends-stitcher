from app import setup_debug_logging
import logging

def test_script_run_context_warning_suppressed():
    setup_debug_logging()
    logger = logging.getLogger("streamlit.runtime.scriptrunner.script_run_context")
    assert logger.getEffectiveLevel() == logging.ERROR
