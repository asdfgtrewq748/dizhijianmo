import logging
import sys
import traceback
from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtWidgets import QMessageBox, QApplication

class QtLogHandler(logging.Handler, QObject):
    """Custom logging handler that emits a signal for each log record."""
    new_record = pyqtSignal(str)

    def __init__(self):
        logging.Handler.__init__(self)
        QObject.__init__(self)

    def emit(self, record):
        try:
            msg = self.format(record)
            self.new_record.emit(msg)
        except Exception:
            self.handleError(record)

def setup_logging(log_file="app.log"):
    """Sets up the logging configuration."""
    for stream in (sys.stdout, sys.stderr):
        try:
            if hasattr(stream, "reconfigure"):
                stream.reconfigure(errors="backslashreplace")
        except Exception:
            pass

    # Create a root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers to avoid duplicates if called multiple times
    if logger.handlers:
        logger.handlers.clear()

    # File handler
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Qt Handler (for UI)
    qt_handler = QtLogHandler()
    qt_formatter = logging.Formatter('%(message)s') # UI usually just needs the message
    qt_handler.setFormatter(qt_formatter)
    logger.addHandler(qt_handler)
    
    # Console handler (optional, for debugging)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(file_formatter)
    logger.addHandler(console_handler)

    return qt_handler

def global_exception_hook(exctype, value, tb):
    """Global exception hook to catch unhandled exceptions."""
    # Log the exception
    logging.critical("Unhandled exception", exc_info=(exctype, value, tb))
    
    # Format the traceback
    error_msg = "".join(traceback.format_exception(exctype, value, tb))
    print(f"Critical Error: {error_msg}") # Ensure it's printed to console too
    
    # Show error message box if QApplication is running
    app = QApplication.instance()
    if app:
        # We need to ensure this runs on the main thread. 
        # Since excepthook is usually called on the thread where exception occurred.
        # If it's the main thread, we can show UI.
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Icon.Critical)
        msg_box.setWindowTitle("Critical Error")
        msg_box.setText(f"An unhandled exception occurred:\n{value}")
        msg_box.setDetailedText(error_msg)
        msg_box.exec()
