# Security

## Overview

FluxFlow UI implements several security measures for local development use. **This application is designed for local use only and should not be exposed to the internet without additional hardening.**

## Implemented Security Measures

### 1. Path Traversal Protection

The `safe_path()` function prevents directory traversal attacks in file operations:

```python
def safe_path(user_path: str, base_dir: str = FILE_BROWSER_BASE_DIR) -> str:
    """Validate and sanitize path to prevent directory traversal.
    
    Args:
        user_path: User-provided path
        base_dir: Base directory to restrict access to
        
    Returns:
        Safe absolute path
        
    Raises:
        ValueError: If path traversal attempt detected
    """
    full_path = os.path.realpath(os.path.join(base_dir, user_path))
    
    if not full_path.startswith(os.path.realpath(base_dir)):
        raise ValueError("Path traversal attempt detected")
    
    return full_path
```

**Implementation:** `src/fluxflow_ui/app_flask.py:52-72`

**Note:** Currently defined but not actively used in all file operations. The file browser uses `_resolve_browse_path()` which does NOT enforce base directory restrictions.

### 2. CORS Configuration

CORS is restricted to localhost only:

```python
CORS(app, origins=["http://localhost:7860", "http://127.0.0.1:7860"])
```

**Implementation:** `src/fluxflow_ui/app_flask.py:29`

**Purpose:** Prevents cross-origin requests from external domains.

### 3. Input Validation

The `@require_json` decorator validates JSON payloads on POST endpoints:

```python
def require_json(f):
    """Decorator to require JSON body in POST requests."""
    @wraps(f)
    def decorated(*args, **kwargs):
        if not request.json:
            return jsonify({"status": "error", "message": "JSON body required"}), 400
        return f(*args, **kwargs)
    return decorated
```

**Implementation:** `src/fluxflow_ui/app_flask.py:40-49`

**Applied to:** All POST endpoints requiring JSON input.

## Known Limitations

### Critical

1. **File Browser Path Traversal Risk**
   - The file browser (`/api/files/browse`) does NOT use `safe_path()` protection
   - Users can browse the entire filesystem (limited by permissions)
   - **Mitigation:** Only run on trusted local machines

2. **No Authentication/Authorization**
   - No user authentication
   - No API key protection
   - Anyone with network access to port 7860 can use the API

3. **No Rate Limiting**
   - Endpoints can be called unlimited times
   - Potential for resource exhaustion

### Moderate

4. **Limited Input Validation**
   - Numeric parameters not bounds-checked
   - String lengths not validated
   - File paths not sanitized before subprocess calls

5. **Subprocess Security**
   - Training runner passes config to subprocess without sanitization
   - Potential for command injection if config contains malicious values

6. **No HTTPS**
   - All communication is unencrypted HTTP
   - Credentials/data transmitted in plaintext

## Production Deployment Warnings

**DO NOT deploy FluxFlow UI to production without addressing:**

1. Add authentication (OAuth2, API keys, or basic auth)
2. Enable HTTPS with valid certificates
3. Implement rate limiting
4. Add comprehensive input validation and sanitization
5. Use `safe_path()` in ALL file operations
6. Run behind a reverse proxy (nginx, Caddy)
7. Use process isolation (containers, VMs)
8. Add audit logging
9. Implement CSP headers
10. Regular security audits

## Recommended Deployment

For local development (current design):
```bash
# Only accessible from localhost
fluxflow-ui
# Access at http://localhost:7860
```

For production (requires additional work):
```bash
# NOT RECOMMENDED - Additional security required
# See warnings above
```

## Reporting Security Issues

Report security vulnerabilities to: danielecamisani@inspiredthinking.group

**Do not** open public issues for security vulnerabilities.

## Test Coverage

Security-related test coverage: **~2%**

Most security features are not covered by automated tests. Manual testing required.
