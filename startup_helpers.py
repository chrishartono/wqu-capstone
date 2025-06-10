import logging
from concurrent_log_handler import ConcurrentRotatingFileHandler

def parallel_logging(name):
	logger = logging.getLogger()
	# Check if handlers are already configured (prevents duplicate handlers)
	if not logger.handlers:
		logger.setLevel(logging.INFO)

		# Set up file handler with concurrency support
		file_handler = ConcurrentRotatingFileHandler(name, mode='a', maxBytes=1024 * 1024, backupCount=5)
		file_format = logging.Formatter('%(asctime)s - %(levelname)s - [%(processName)s] - %(message)s')
		file_handler.setFormatter(file_format)
		logger.addHandler(file_handler)

		# Set up console handler
		console_handler = logging.StreamHandler()
		console_format = logging.Formatter('%(asctime)s - %(levelname)s - [%(processName)s] - %(message)s')
		console_handler.setFormatter(console_format)
		logger.addHandler(console_handler)

	return logger

def IsLoggingConfigured():
	return logging.getLogger().hasHandlers()

def SetLogging(logname: str, append: bool = False):
	mode = 'a' if append else 'w'
	logging.basicConfig(format='%(asctime)s.%(msecs)03d;%(levelname)s;{%(module)s};[%(funcName)s];%(thread)d-%(process)d;%(message)s',
						datefmt='%d/%m/%Y %I:%M:%S',
						handlers=[logging.StreamHandler(), logging.FileHandler(logname, mode=mode)],
						level=logging.INFO)

def ResetLogFileHandler(logname: str):
	logger = logging.getLogger()
	logger.handlers[1].stream.close()
	logger.removeHandler(logger.handlers[1])

	file_handler = logging.FileHandler(logname)
	file_handler.setLevel(logging.INFO)
	formatter = logging.Formatter('%(asctime)s.%(msecs)03d;%(levelname)s;{%(module)s};[%(funcName)s];%(message)s')
	file_handler.setFormatter(formatter)
	logger.addHandler(file_handler)