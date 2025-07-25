class AiHandlerError(Exception):
    """Base class for all exceptions raised by the AI handler library."""

    pass


class ClientError(AiHandlerError):
    """Base class for client-related errors."""

    pass


class ProviderError(ClientError):
    """Raised when there is an error with the AI provider."""

    pass


class InvalidModelResponseException(ClientError):
    """Raised when the AI model returns an invalid response."""

    def __init__(self, origin: str | Exception, message: str = None):
        self.origin = origin
        super().__init__(message or "The AI model returned an invalid response.")
