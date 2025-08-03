class BaseService:
    """
    Base class for all services in the Prometheus application.
    """

    def close(self):
        """
        Close the service and release any resources.
        This method should be overridden by subclasses to implement specific cleanup logic.
        """
        pass
