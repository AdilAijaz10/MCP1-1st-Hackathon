
import asyncio
import json
import io
import base64
from typing import Any
from mcp.server import Server
from mcp.types import Tool, TextContent
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

# Initialize MCP server
server = Server("math-mcp-server")

# Helper utilities ----------------------------------------------------------
def _to_json_serializable(o):
	# Convert sympy/ numpy types to JSON-serializable forms
	try:
		if isinstance(o, (sp.Basic,)):
			try:
				return float(sp.N(o))
			except Exception:
				return str(o)
		if isinstance(o, (np.ndarray,)):
			return o.tolist()
		return o
	except Exception:
		return str(o)

def _format_result(payload: dict) -> str:
	# Convert all Sympy/numpy values into serializable form recursively
	def conv(x):
		if isinstance(x, dict):
			return {k: conv(v) for k, v in x.items()}
		if isinstance(x, list):
			return [conv(v) for v in x]
		return _to_json_serializable(x)
	return json.dumps(conv(payload), indent=2)

# Core math operations -----------------------------------------------------
def solve_equation_proc(equation: str, variable: str = "x"):
	try:
		var = sp.symbols(variable)
		if "=" in equation:
			lhs, rhs = equation.split("=", 1)
			expr = sp.sympify(lhs) - sp.sympify(rhs)
		else:
			expr = sp.sympify(equation)
		sol = sp.solve(sp.Eq(expr, 0), var)
		sol_n = [sp.N(s) if not s.free_symbols else s for s in sol]
		payload = {
			"input": equation,
			"variable": str(var),
			"solutions": [str(s) for s in sol],
			"numeric_solutions": [str(s) for s in sol_n],
			"count": len(sol)
		}
	except Exception as e:
		payload = {"error": f"Failed to solve equation: {str(e)}"}
	return payload

def integrate_proc(expression: str, variable: str = "x", lower=None, upper=None):
	try:
		var = sp.symbols(variable)
		expr = sp.sympify(expression)
		if lower is not None and upper is not None and lower != "" and upper != "":
			lower_v = sp.sympify(lower)
			upper_v = sp.sympify(upper)
			res = sp.integrate(expr, (var, lower_v, upper_v))
			numeric = sp.N(res)
			payload = {
				"input": expression,
				"variable": variable,
				"bounds": [str(lower_v), str(upper_v)],
				"result": str(res),
				"numeric_result": str(numeric)
			}
		else:
			res = sp.integrate(expr, var)
			payload = {
				"input": expression,
				"variable": variable,
				"result": str(res)
			}
	except Exception as e:
		payload = {"error": f"Integration failed: {str(e)}"}
	return payload

def differentiate_proc(expression: str, variable: str = "x", order: int = 1):
	try:
		var = sp.symbols(variable)
		expr = sp.sympify(expression)
		res = sp.diff(expr, var, order)
		payload = {
			"input": expression,
			"variable": variable,
			"order": order,
			"derivative": str(res)
		}
	except Exception as e:
		payload = {"error": f"Differentiation failed: {str(e)}"}
	return payload

def limit_proc(expression: str, variable: str = "x", point="oo", dir="+"):
	try:
		var = sp.symbols(variable)
		expr = sp.sympify(expression)
		if isinstance(point, str):
			if point == "oo":
				pt = sp.oo
			elif point == "-oo":
				pt = -sp.oo
			else:
				pt = sp.sympify(point)
		else:
			pt = sp.sympify(point)
		res = sp.limit(expr, var, pt, dir=dir)
		payload = {
			"input": expression,
			"variable": variable,
			"point": str(pt),
			"direction": dir,
			"limit": str(res)
		}
	except Exception as e:
		payload = {"error": f"Limit computation failed: {str(e)}"}
	return payload

def plot_function_proc(expression: str, variable: str = "x", start: float = -10.0, end: float = 10.0, points: int = 1000):
	try:
		var = sp.symbols(variable)
		expr = sp.sympify(expression)
		f = sp.lambdify(var, expr, modules=["numpy", "sympy"])
		xs = np.linspace(start, end, points)
		with np.errstate(all='ignore'):
			ys = f(xs)
		ys = np.array(ys, dtype=np.float64)
		fig, ax = plt.subplots(figsize=(6, 3.5))
		ax.plot(xs, ys, linewidth=1)
		ax.set_title(f"y = {str(expr)}")
		ax.grid(True)
		ax.set_xlabel(variable)
		ax.set_ylabel("y")
		buf = io.BytesIO()
		fig.savefig(buf, format="png", bbox_inches='tight')
		plt.close(fig)
		buf.seek(0)
		img_b64 = base64.b64encode(buf.getvalue()).decode("ascii")
		payload = {
			"input": expression,
			"variable": variable,
			"range": [start, end],
			"points": points,
			"plot_image_base64": img_b64
		}
	except Exception as e:
		payload = {"error": f"Plotting failed: {str(e)}"}
	return payload

def simplify_evaluate_proc(expression: str, evaluate: bool = False, subs: dict = None):
	try:
		expr = sp.sympify(expression)
		simplified = sp.simplify(expr)
		payload = {
			"input": expression,
			"simplified": str(simplified)
		}
		if evaluate:
			if subs:
				subs_map = {sp.symbols(k): sp.sympify(v) for k, v in subs.items()}
				numeric = sp.N(simplified.subs(subs_map))
			else:
				numeric = sp.N(simplified)
			payload["numeric"] = str(numeric)
	except Exception as e:
		payload = {"error": f"Simplify/Evaluate failed: {str(e)}"}
	return payload

def matrix_operation_proc(matrix, operation: str = "inverse", other=None):
	try:
		M = sp.Matrix(matrix)
		result = None
		if operation == "inverse":
			result = M.inv()
		elif operation == "determinant":
			result = M.det()
		elif operation == "transpose":
			result = M.T
		elif operation == "eigen":
			ev = M.eigenvals()
			result = {str(k): int(v) for k, v in ev.items()}
		elif operation == "multiply":
			if other is None:
				raise ValueError("Missing multiplier")
			if isinstance(other, list):
				result = M * sp.Matrix(other)
			else:
				result = M * sp.sympify(other)
		else:
			raise ValueError(f"Unsupported operation: {operation}")
		payload = {
			"operation": operation,
			"input_matrix": [[str(v) for v in row] for row in M.tolist()],
			"result": str(result)
		}
	except Exception as e:
		payload = {"error": f"Matrix operation failed: {str(e)}"}
	return payload

def latexify_proc(expression: str):
	try:
		expr = sp.sympify(expression)
		latex = sp.latex(expr)
		payload = {"input": expression, "latex": latex}
	except Exception as e:
		payload = {"error": f"LaTeX conversion failed: {str(e)}"}
	return payload

def batch_process_proc(documents: list):
	results = []
	for doc in documents:
		op = doc.get("operation")
		payload = {"id": doc.get("id", None), "operation": op}
		try:
			if op == "solve":
				payload["result"] = solve_equation_proc(doc["expression"], doc.get("variable", "x"))
			elif op == "integrate":
				payload["result"] = integrate_proc(doc["expression"], doc.get("variable", "x"), doc.get("lower"), doc.get("upper"))
			elif op == "differentiate":
				payload["result"] = differentiate_proc(doc["expression"], doc.get("variable", "x"), doc.get("order", 1))
			elif op == "limit":
				payload["result"] = limit_proc(doc["expression"], doc.get("variable", "x"), doc.get("point", "oo"), doc.get("dir", "+"))
			elif op == "plot":
				payload["result"] = plot_function_proc(doc["expression"], doc.get("variable", "x"), doc.get("start", -10), doc.get("end", 10), doc.get("points", 1000))
			elif op == "simplify":
				payload["result"] = simplify_evaluate_proc(doc["expression"], doc.get("evaluate", False), doc.get("subs"))
			elif op == "matrix":
				payload["result"] = matrix_operation_proc(doc["matrix"], doc.get("matrix_op", "inverse"), doc.get("other"))
			elif op == "latex":
				payload["result"] = latexify_proc(doc["expression"])
			else:
				payload["error"] = f"Unknown operation: {op}"
		except Exception as e:
			payload["error"] = str(e)
		results.append(payload)
	return {"total": len(results), "results": results}

# MCP tool listing ---------------------------------------------------------
@server.list_tools()
async def list_tools() -> list[Tool]:
	return [
		Tool(
			name="solve_equation",
			description="Solve algebraic equations (symbolic and numeric).",
			inputSchema={
				"type": "object",
				"properties": {
					"equation": {"type": "string"},
					"variable": {"type": "string", "default": "x"}
				},
				"required": ["equation"]
			}
		),
		Tool(
			name="integrate",
			description="Symbolic integration (definite and indefinite).",
			inputSchema={
				"type": "object",
				"properties": {
					"expression": {"type": "string"},
					"variable": {"type": "string", "default": "x"},
					"lower": {"type": ["string", "number"], "default": None},
					"upper": {"type": ["string", "number"], "default": None}
				},
				"required": ["expression"]
			}
		),
		Tool(
			name="differentiate",
			description="Symbolic differentiation.",
			inputSchema={
				"type": "object",
				"properties": {
					"expression": {"type": "string"},
					"variable": {"type": "string", "default": "x"},
					"order": {"type": "integer", "default": 1}
				},
				"required": ["expression"]
			}
		),
		Tool(
			name="limit",
			description="Compute limits (including l'Hôpital when appropriate).",
			inputSchema={
				"type": "object",
				"properties": {
					"expression": {"type": "string"},
					"variable": {"type": "string", "default": "x"},
					"point": {"type": "string", "default": "oo"},
					"dir": {"type": "string", "enum": ["+", "-"], "default": "+"}
				},
				"required": ["expression"]
			}
		),
		Tool(
			name="plot_function",
			description="Plot functions; returns base64 PNG in JSON.",
			inputSchema={
				"type": "object",
				"properties": {
					"expression": {"type": "string"},
					"variable": {"type": "string", "default": "x"},
					"start": {"type": "number", "default": -10},
					"end": {"type": "number", "default": 10},
					"points": {"type": "integer", "default": 1000}
				},
				"required": ["expression"]
			}
		),
		Tool(
			name="simplify_evaluate",
			description="Simplify expressions and optionally evaluate numerically with substitutions.",
			inputSchema={
				"type": "object",
				"properties": {
					"expression": {"type": "string"},
					"evaluate": {"type": "boolean", "default": False},
					"subs": {"type": "object"}
				},
				"required": ["expression"]
			}
		),
		Tool(
			name="matrix_operation",
			description="Matrix operations: inverse, determinant, transpose, eigen, multiply.",
			inputSchema={
				"type": "object",
				"properties": {
					"matrix": {"type": "array"},
					"matrix_op": {"type": "string", "enum": ["inverse", "determinant", "transpose", "eigen", "multiply"], "default": "inverse"},
					"other": {"type": ["array", "number", "string"]}
				},
				"required": ["matrix"]
			}
		),
		Tool(
			name="latexify",
			description="Return LaTeX representation of an expression.",
			inputSchema={
				"type": "object",
				"properties": {"expression": {"type": "string"}},
				"required": ["expression"]
			}
		),
		Tool(
			name="batch_process",
			description="Batch multiple operations.",
			inputSchema={
				"type": "object",
				"properties": {
					"documents": {"type": "array"}
				},
				"required": ["documents"]
			}
		)
	]

# MCP call handler --------------------------------------------------------
@server.call_tool()
async def call_tool(name: str, arguments: Any) -> list[TextContent]:
	try:
		if name == "solve_equation":
			out = solve_equation_proc(arguments["equation"], arguments.get("variable", "x"))
		elif name == "integrate":
			out = integrate_proc(arguments["expression"], arguments.get("variable", "x"), arguments.get("lower"), arguments.get("upper"))
		elif name == "differentiate":
			out = differentiate_proc(arguments["expression"], arguments.get("variable", "x"), arguments.get("order", 1))
		elif name == "limit":
			out = limit_proc(arguments["expression"], arguments.get("variable", "x"), arguments.get("point", "oo"), arguments.get("dir", "+"))
		elif name == "plot_function":
			out = plot_function_proc(arguments["expression"], arguments.get("variable", "x"), float(arguments.get("start", -10)), float(arguments.get("end", 10)), int(arguments.get("points", 1000)))
		elif name == "simplify_evaluate":
			out = simplify_evaluate_proc(arguments["expression"], arguments.get("evaluate", False), arguments.get("subs"))
		elif name == "matrix_operation":
			out = matrix_operation_proc(arguments["matrix"], arguments.get("matrix_op", "inverse"), arguments.get("other"))
		elif name == "latexify":
			out = latexify_proc(arguments["expression"])
		elif name == "batch_process":
			out = batch_process_proc(arguments["documents"])
		else:
			out = {"error": f"Unknown tool: {name}"}
	except Exception as e:
		out = {"error": str(e)}
	return [TextContent(type="text", text=_format_result(out))]

# Run server (stdio wrapper for MCP) ---------------------------------------
async def main():
	async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
		await server.run(
			read_stream,
			write_stream,
			server.create_initialization_options()
		)

if __name__ == "__main__":
	asyncio.run(main())