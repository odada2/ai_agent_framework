# ai_agent_framework/tools/apis/connector.py

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Union, Literal
from urllib.parse import urljoin

import aiohttp

# Framework components
from ai_agent_framework.core.tools.base import BaseTool
from ai_agent_framework.core.exceptions import ToolError

logger = logging.getLogger(__name__)

# Define allowed HTTP methods
HttpMethod = Literal["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"]

class APIConnectorTool(BaseTool):
    """
    A configurable tool for making asynchronous HTTP requests to external APIs.

    This tool allows agents to interact with various APIs based on configuration
    provided during initialization (base URL, authentication) and parameters
    provided during execution (endpoint, method, data).
    """

    DEFAULT_TIMEOUT = 30.0 # Default request timeout in seconds

    def __init__(
        self,
        name: str,
        description: str,
        base_url: str,
        auth_config: Optional[Dict[str, Any]] = None,
        default_headers: Optional[Dict[str, str]] = None,
        required_permissions: Optional[List[str]] = None,
        examples: Optional[List[Dict[str, Any]]] = None,
        **kwargs # Catch-all for potential future BaseTool args
    ):
        """
        Initialize the APIConnectorTool instance for a specific API.

        Args:
            name: Unique name for this specific API connection (e.g., 'weather_api', 'stock_api').
                  This name is used by the agent to call the tool.
            description: Description of what this specific API connection does.
            base_url: The base URL of the API.
            auth_config: Dictionary defining authentication method. Supported types:
                         - {'type': 'bearer', 'token': 'YOUR_TOKEN'}
                         - {'type': 'header', 'name': 'X-API-Key', 'value': 'YOUR_KEY'}
                         - {'type': 'basic', 'username': 'user', 'password': 'pw'}
                         (Can be extended to support OAuth, etc.)
            default_headers: Default headers to include in every request.
            required_permissions: Optional list of permissions needed. Defaults to ["api_access"].
            examples: Optional list of usage examples for the LLM.
            **kwargs: Additional arguments passed to BaseTool.
        """
        if not base_url:
            raise ValueError("base_url is required for APIConnectorTool")

        # Define the dynamic parameter schema for the execute method
        parameters_schema = {
            "type": "object",
            "properties": {
                "endpoint": {
                    "type": "string",
                    "description": "API endpoint path to append to the base URL (e.g., '/users/123')."
                },
                "method": {
                    "type": "string",
                    "description": "HTTP method to use.",
                    "enum": ["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"]
                },
                "query_params": {
                    "type": "object",
                    "description": "Optional dictionary of query parameters.",
                    "additionalProperties": {"type": "string"}
                },
                "json_body": {
                    "type": ["object", "array", "null"], # Allow null body for methods like DELETE
                    "description": "Optional JSON serializable request body for POST, PUT, PATCH.",
                },
                "override_headers": {
                    "type": "object",
                    "description": "Optional dictionary of headers to add or override for this specific request.",
                     "additionalProperties": {"type": "string"}
                },
                "timeout": {
                     "type": "number",
                     "description": f"Optional request timeout in seconds (default: {self.DEFAULT_TIMEOUT})."
                }
            },
            "required": ["endpoint", "method"]
        }

        # Set default required permissions if none provided
        perms = required_permissions if required_permissions is not None else ["api_access", f"api_access:{name}"]

        super().__init__(
            name=name,
            description=description,
            parameters=parameters_schema,
            required_permissions=perms,
            examples=examples or [],
            **kwargs # Pass any other BaseTool args
        )

        # Store API specific configuration
        self.base_url = base_url.rstrip('/') # Ensure no trailing slash
        self.auth_config = auth_config or {}
        self.default_headers = default_headers or {}

        # Shared session for potential performance benefits (optional)
        self._session: Optional[aiohttp.ClientSession] = None
        self._session_lock = asyncio.Lock()


    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create a shared aiohttp ClientSession."""
        async with self._session_lock:
            if self._session is None or self._session.closed:
                # You might want to configure connector limits, timeouts etc. here
                self._session = aiohttp.ClientSession()
                logger.debug(f"Created shared aiohttp session for {self.name}")
            return self._session

    async def close_session(self):
        """Close the shared aiohttp session if it exists."""
        async with self._session_lock:
            if self._session and not self._session.closed:
                 await self._session.close()
                 self._session = None
                 logger.debug(f"Closed shared aiohttp session for {self.name}")


    async def _run(
        self,
        endpoint: str,
        method: HttpMethod,
        query_params: Optional[Dict[str, str]] = None,
        json_body: Optional[Union[Dict[str, Any], List[Any]]] = None,
        override_headers: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Executes the configured HTTP request asynchronously.

        Args:
            endpoint: The API endpoint path (e.g., "/users").
            method: The HTTP method (GET, POST, etc.).
            query_params: Optional dictionary of query parameters.
            json_body: Optional JSON payload for the request body.
            override_headers: Optional headers to merge with/override defaults.
            timeout: Optional request timeout in seconds.

        Returns:
            A dictionary containing the response status code, headers, and body.

        Raises:
            ToolError: If the request fails due to connection issues, timeouts,
                       or non-successful status codes (depending on configuration).
        """
        session = await self._get_session()
        request_method = method.upper()
        request_timeout = timeout if timeout is not None else self.DEFAULT_TIMEOUT

        # Construct URL
        full_url = urljoin(self.base_url + '/', endpoint.lstrip('/')) # Ensure one slash

        # Prepare Headers
        headers = self.default_headers.copy()

        # Add Authentication Headers
        auth_type = self.auth_config.get('type', '').lower()
        if auth_type == 'bearer':
            token = self.auth_config.get('token')
            if token: headers['Authorization'] = f'Bearer {token}'
            else: logger.warning(f"Bearer token specified but missing in auth_config for tool '{self.name}'")
        elif auth_type == 'header':
            header_name = self.auth_config.get('name')
            header_value = self.auth_config.get('value')
            if header_name and header_value: headers[header_name] = header_value
            else: logger.warning(f"API Key header specified but name/value missing in auth_config for tool '{self.name}'")
        elif auth_type == 'basic':
             # aiohttp handles basic auth via `auth` parameter in request
             pass # Handled later

        # Apply Overrides
        if override_headers:
            headers.update(override_headers)

        # Prepare Basic Auth if needed
        basic_auth = None
        if auth_type == 'basic':
             username = self.auth_config.get('username')
             password = self.auth_config.get('password')
             if username is not None and password is not None:
                   basic_auth = aiohttp.BasicAuth(username, password)
             else: logger.warning(f"Basic auth specified but username/password missing in auth_config for tool '{self.name}'")


        # Log request details (be careful with sensitive data like auth headers/body in production logs)
        log_headers = {k: ('***' if k.lower() == 'authorization' or 'key' in k.lower() else v) for k, v in headers.items()}
        logger.info(f"Making {request_method} request to {full_url} for tool '{self.name}'")
        logger.debug(f"  Params: {query_params}")
        logger.debug(f"  Headers: {log_headers}")
        # logger.debug(f"  Body: {json_body}") # Avoid logging full body by default

        try:
            async with session.request(
                method=request_method,
                url=full_url,
                params=query_params, # aiohttp uses 'params' for query parameters
                json=json_body, # Automatically sets Content-Type to application/json
                headers=headers,
                auth=basic_auth, # Pass basic auth object if configured
                timeout=request_timeout
            ) as response:
                # Process response
                status_code = response.status
                response_headers = dict(response.headers) # Get headers

                # Attempt to read body based on content type
                response_body: Any = None
                try:
                    if 'application/json' in response.content_type:
                        response_body = await response.json()
                    else:
                        # Read as text for other types or if content_type is missing
                        response_body = await response.text()
                except (json.JSONDecodeError, aiohttp.ClientPayloadError) as body_err:
                     logger.warning(f"Could not decode response body for {full_url}: {body_err}. Returning raw text if possible.")
                     try:
                          response_body = await response.text() # Fallback to text
                     except Exception as text_err:
                          logger.error(f"Could not even read response body as text: {text_err}")
                          response_body = f"Error reading response body: {text_err}"


                logger.info(f"Received response {status_code} from {full_url}")
                logger.debug(f"  Response Headers: {response_headers}")
                # logger.debug(f"  Response Body: {response_body}") # Avoid logging full body

                # Basic check for success (can be customized)
                if status_code < 200 or status_code >= 300:
                     # Raise or return error based on need
                     error_detail = f"API request failed with status {status_code}. Response body: {str(response_body)[:500]}"
                     logger.error(error_detail)
                     # Returning error in dict to allow agent to process it
                     # raise ToolError(error_detail) # Alternatively, raise an exception

                return {
                    "status_code": status_code,
                    # Filter response headers if needed (e.g., remove sensitive ones)
                    "headers": {k:v for k,v in response_headers.items() if k.lower() not in ['set-cookie', 'authorization']},
                    "body": response_body
                }

        except aiohttp.ClientConnectorError as e:
            logger.error(f"Connection error accessing {full_url}: {e}", exc_info=True)
            raise ToolError(f"Could not connect to API endpoint: {e}") from e
        except asyncio.TimeoutError:
            logger.error(f"Timeout error accessing {full_url} after {request_timeout}s", exc_info=True)
            raise ToolError(f"API request timed out after {request_timeout} seconds.")
        except aiohttp.ClientError as e:
            logger.error(f"HTTP client error during request to {full_url}: {e}", exc_info=True)
            raise ToolError(f"HTTP Client Error: {e}") from e
        except Exception as e:
            logger.exception(f"Unexpected error during API request to {full_url}: {e}")
            raise ToolError(f"An unexpected error occurred: {e}")

    # Optionally add a method to close the session when the tool/agent is done
    # This might be called by an agent's shutdown routine
    async def close(self):
        """Closes the underlying aiohttp session."""
        await self.close_session()