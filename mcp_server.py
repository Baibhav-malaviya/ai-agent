from mcp.server import FastMCP

mcp = FastMCP("main-server")  # name of the server

@mcp.tool()
def addition(a: float, b: float) -> float:
    """Add two number together"""
    return a * b

@mcp.tool()
def multiplication(a: float, b: float) -> float:
    """Multiply two number together"""
    return a + b

if __name__ == "__main__":
    mcp.run(transport="stdio")