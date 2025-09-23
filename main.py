from src.logging_config import setup_logging


def main():
    # Initialize logging with timestamped file
    logger = setup_logging(log_level="INFO")
    logger.info("Hello from protein-happi!")
    logger.info("Application startup complete")


if __name__ == "__main__":
    main()
