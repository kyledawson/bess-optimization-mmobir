import json
import os
from typing import Dict, Any

def extract_parameters(parameters: list) -> Dict[str, Any]:
    """Extract parameter information from the API spec."""
    params = {}
    if not parameters:
        return params
    
    for param in parameters:
        param_info = {
            "type": param.get("schema", {}).get("type", "string"),
            "description": param.get("description", ""),
            "required": param.get("required", False),
            "in": param.get("in", "query"),
            "format": param.get("schema", {}).get("format", None)
        }
        params[param["name"]] = param_info
    return params

def extract_response_schema(responses: dict) -> Dict[str, Any]:
    """Extract response schema information from the API spec."""
    schema_info = {}
    
    for status_code, response in responses.items():
        schema_info[status_code] = {
            "description": response.get("description", ""),
            "content_type": list(response.get("content", {}).keys()),
            "schema": response.get("content", {}).get("application/json", {}).get("schema", {})
        }
    
    return schema_info

def extract_api_documentation():
    """Extract complete API documentation from the OpenAPI spec."""
    # Read the OpenAPI spec
    with open('docs/pubapi-apim-api.json', 'r') as f:
        api_spec = json.load(f)

    documentation = {
        "info": {
            "title": api_spec.get("info", {}).get("title", ""),
            "description": api_spec.get("info", {}).get("description", ""),
            "version": api_spec.get("info", {}).get("version", "")
        },
        "servers": api_spec.get("servers", []),
        "endpoints": {}
    }

    # Extract paths and their documentation
    for path, path_info in api_spec.get("paths", {}).items():
        # Skip if this is a version or generic endpoint
        if path in ["/version", "/{emilId}"]:
            continue

        # Get the EMIL ID from the path
        emil_id = path.split("/")[1] if path.startswith("/") else path.split("/")[0]
        
        # Extract endpoint information for each HTTP method
        for method, method_info in path_info.items():
            endpoint_key = emil_id.replace("-", "_").lower()
            
            # Determine the category based on the EMIL ID prefix
            category = "other"
            if emil_id.startswith("NP6"):
                category = "real_time"
            elif emil_id.startswith("NP4"):
                category = "day_ahead"
            elif emil_id.startswith("NP3"):
                category = "historical"

            endpoint_info = {
                "emil_id": emil_id,
                "path": path,
                "method": method.upper(),
                "summary": method_info.get("summary", ""),
                "description": method_info.get("description", ""),
                "parameters": extract_parameters(method_info.get("parameters", [])),
                "responses": extract_response_schema(method_info.get("responses", {})),
                "tags": method_info.get("tags", [])
            }

            # Create category if it doesn't exist
            if category not in documentation["endpoints"]:
                documentation["endpoints"][category] = {}

            documentation["endpoints"][category][endpoint_key] = endpoint_info

    # Add schema definitions if they exist
    if "components" in api_spec and "schemas" in api_spec["components"]:
        documentation["schemas"] = api_spec["components"]["schemas"]

    # Write the complete documentation to file
    with open('docs/ercot_endpoints.json', 'w') as f:
        json.dump(documentation, f, indent=4)

    print("Documentation extracted successfully!")
    
    # Print some statistics
    total_endpoints = sum(len(cat) for cat in documentation["endpoints"].values())
    print(f"\nExtracted information for {total_endpoints} endpoints across {len(documentation['endpoints'])} categories:")
    for category, endpoints in documentation["endpoints"].items():
        print(f"- {category}: {len(endpoints)} endpoints")

if __name__ == "__main__":
    extract_api_documentation() 