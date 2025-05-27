import logging
from functools import wraps
from typing import Any, Callable, TypeVar, cast

#from quart import abort, current_app, request
from fastapi import Request, HTTPException
from config import CONFIG_AUTH_CLIENT, CONFIG_SEARCH_CLIENT
from core.authentication import AuthError
from error import error_response
from config import CONFIG_AUTH_CLIENT, CONFIG_SEARCH_CLIENT

def authenticated_path(route_fn: Callable[[str, dict[str, Any]], Any]):
    """
    Decorator for routes that request a specific file that might require access control enforcement
    """

    @wraps(route_fn)
    async def auth_handler(path: str, request: Request, *args, **kwargs):
        # If authentication is enabled, validate the user can access the file
        # auth_helper = current_app.config[CONFIG_AUTH_CLIENT]
        # search_client = current_app.config[CONFIG_SEARCH_CLIENT]
        auth_helper = getattr(request.app.state, CONFIG_AUTH_CLIENT)
        search_client = getattr(request.app.state, CONFIG_SEARCH_CLIENT)

        authorized = False
        try:
            auth_claims = await auth_helper.get_auth_claims_if_enabled(request.headers)
            authorized = await auth_helper.check_path_auth(path, auth_claims, search_client)
        except AuthError:
            raise HTTPException(status_code=403, detail="Forbidden")
        except Exception as error:
            logging.exception("Problem checking path auth %s", error)
            return error_response(error, route="/content")

        if not authorized:
            raise HTTPException(status_code=403, detail="Forbidden")

        return await route_fn(path, auth_claims)

    return auth_handler


_C = TypeVar("_C", bound=Callable[..., Any])


def authenticated(route_fn: _C) -> _C:
    """
    Decorator for routes that might require access control. Unpacks Authorization header information into an auth_claims dictionary
    """

    @wraps(route_fn)
    async def auth_handler(request: Request, *args, **kwargs):
        auth_helper = getattr(request.app.state, CONFIG_AUTH_CLIENT)
        try:
            auth_claims = await auth_helper.get_auth_claims_if_enabled(request.headers)
        except AuthError:
            raise HTTPException(status_code=403, detail="Forbidden")

        return await route_fn(request, *args, **kwargs)

    return cast(_C, auth_handler)
