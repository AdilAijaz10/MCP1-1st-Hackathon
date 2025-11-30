import gradio as gr
import json
import base64
from typing import Tuple
from PIL import Image
import io
import mcp_server

EXAMPLES = {
	"solve": "x**3 - 6*x**2 + 11*x - 6 = 0",
	"integrate": "exp(-x**2)",
	"differentiate": "sin(x)*cos(x)",
	"limit": "(1 + 1/n)**n",
	"plot": "sin(x)/x",
	"matrix": "[[1,2],[3,4]]",
	"simplify": "sin(x)**2 + cos(x)**2",
	"latex": "Integral(exp(-x**2), (x, -oo, oo))"
}

TOOL_DESCRIPTIONS = {
	"solve": {
		"title": "Solve Equation",
		"desc": "Find symbolic and numeric solutions to algebraic equations.",
		"usage": "Enter an equation like: x**2 - 5*x + 6 = 0"
	},
	"integrate": {
		"title": "Integrate",
		"desc": "Compute indefinite or definite integrals of expressions.",
		"usage": "Enter an expression like: 1/(x**2 + 1)"
	},
	"differentiate": {
		"title": "Differentiate",
		"desc": "Compute derivatives of any order for expressions.",
		"usage": "Enter an expression like: x**3 + 2*x**2 - 5*x + 1"
	},
	"limit": {
		"title": "Compute Limit",
		"desc": "Find limits of expressions as variables approach values or infinity.",
		"usage": "Enter an expression like: sin(x)/x"
	},
	"plot": {
		"title": "Plot Function",
		"desc": "Visualize functions with customizable range and resolution.",
		"usage": "Enter a function like: sin(x)*cos(x)"
	},
	"simplify": {
		"title": "Simplify Expression",
		"desc": "Simplify mathematical expressions and evaluate them numerically.",
		"usage": "Enter an expression like: (x**2 - 1)/(x - 1)"
	},
	"matrix": {
		"title": "Matrix Operation",
		"desc": "Perform operations: inverse, determinant, transpose, eigenvalues.",
		"usage": "Enter a matrix like: [[1,2],[3,4]]"
	},
	"latex": {
		"title": "LaTeX Output",
		"desc": "Convert mathematical expressions to LaTeX format.",
		"usage": "Enter an expression like: Integral(exp(-x**2), (x, -oo, oo))"
	}
}

def call_helper(tool: str, inputs: dict) -> Tuple[str, str, object]:
	"""Call the corresponding helper in mcp_server and return (markdown, json_str, image_pil)"""
	try:
		if tool == "solve":
			out = mcp_server.solve_equation_proc(inputs["equation"], inputs.get("variable", "x"))
		elif tool == "integrate":
			out = mcp_server.integrate_proc(inputs["expression"], inputs.get("variable", "x"), inputs.get("lower"), inputs.get("upper"))
		elif tool == "differentiate":
			out = mcp_server.differentiate_proc(inputs["expression"], inputs.get("variable", "x"), int(inputs.get("order", 1)))
		elif tool == "limit":
			out = mcp_server.limit_proc(inputs["expression"], inputs.get("variable", "n"), inputs.get("point", "oo"), inputs.get("dir", "+"))
		elif tool == "plot":
			out = mcp_server.plot_function_proc(inputs["expression"], inputs.get("variable", "x"), float(inputs.get("start", -20)), float(inputs.get("end", 20)), int(inputs.get("points", 1000)))
		elif tool == "simplify":
			out = mcp_server.simplify_evaluate_proc(inputs["expression"], inputs.get("evaluate", False), inputs.get("subs"))
		elif tool == "matrix":
			out = mcp_server.matrix_operation_proc(inputs["matrix"], inputs.get("matrix_op", "inverse"), inputs.get("other"))
		elif tool == "latex":
			out = mcp_server.latexify_proc(inputs["expression"])
		else:
			out = {"error": "Unknown tool"}
	except Exception as e:
		out = {"error": str(e)}
	
	json_out = json.dumps(out, indent=2)
	md = "### ✅ Result\n"
	img_pil = None
	
	if "error" in out:
		md += f"❌ **Error:** {out['error']}\n"
		return md, json_out, img_pil
	
	if "solutions" in out:
		md += f"**🔍 Solutions Found:** {len(out['solutions'])}\n"
		for i, sol in enumerate(out['solutions'], 1):
			md += f"  - x_{i} = {sol}\n"
	if "derivative" in out:
		md += f"**d/dx = ** `{out['derivative']}`\n"
	if "limit" in out:
		md += f"**lim = ** `{out['limit']}`\n"
	if "simplified" in out:
		md += f"**Simplified:** `{out['simplified']}`\n"
	if "latex" in out:
		md += f"**LaTeX:** \n```latex\n{out['latex']}\n```\n"
	if "result" in out and isinstance(out["result"], str):
		md += f"**Result:** `{out['result']}`\n"
	if "numeric_result" in out:
		md += f"**Numeric:** `{out['numeric_result']}`\n"
	if "numeric" in out:
		md += f"**Value:** `{out['numeric']}`\n"
	
	# Handle base64 image (for plot)
	if "plot_image_base64" in out:
		try:
			b64_str = out["plot_image_base64"]
			img_bytes = base64.b64decode(b64_str)
			img_pil = Image.open(io.BytesIO(img_bytes))
			md += "\n**📊 Plot Preview:**\n"
		except Exception as e:
			md += f"\n⚠️ Could not decode plot image: {str(e)}\n"
	
	return md, json_out, img_pil

def update_ui(tool_name):
	"""Return visibility updates for each input field based on selected tool"""
	visibility = {
		"solve": {"expr": True, "var": True, "lower": False, "upper": False, "order": False, "start": False, "end": False, "points": False, "mop": False, "other": False},
		"integrate": {"expr": True, "var": True, "lower": True, "upper": True, "order": False, "start": False, "end": False, "points": False, "mop": False, "other": False},
		"differentiate": {"expr": True, "var": True, "lower": False, "upper": False, "order": True, "start": False, "end": False, "points": False, "mop": False, "other": False},
		"limit": {"expr": True, "var": True, "lower": False, "upper": False, "order": False, "start": False, "end": False, "points": False, "mop": False, "other": True},
		"plot": {"expr": True, "var": True, "lower": False, "upper": False, "order": False, "start": True, "end": True, "points": True, "mop": False, "other": False},
		"simplify": {"expr": True, "var": False, "lower": False, "upper": False, "order": False, "start": False, "end": False, "points": False, "mop": False, "other": True},
		"matrix": {"expr": True, "var": False, "lower": False, "upper": False, "order": False, "start": False, "end": False, "points": False, "mop": True, "other": True},
		"latex": {"expr": True, "var": False, "lower": False, "upper": False, "order": False, "start": False, "end": False, "points": False, "mop": False, "other": False},
	}
	v = visibility.get(tool_name, {})
	
	# Get tool description
	tool_info = TOOL_DESCRIPTIONS.get(tool_name, {})
	info_text = f"### {tool_info.get('title', 'Tool')}\n{tool_info.get('desc', '')}\n\n**Example:** `{tool_info.get('usage', '')}`"
	
	return (
		gr.update(visible=v.get("expr", False), label=_get_input_label(tool_name)),
		gr.update(visible=v.get("var", False)),
		gr.update(visible=v.get("lower", False)),
		gr.update(visible=v.get("upper", False)),
		gr.update(visible=v.get("order", False)),
		gr.update(visible=v.get("start", False)),
		gr.update(visible=v.get("end", False)),
		gr.update(visible=v.get("points", False)),
		gr.update(visible=v.get("mop", False)),
		gr.update(visible=v.get("other", False)),
		gr.update(value=info_text)
	)

def _get_input_label(tool_name: str) -> str:
	"""Return appropriate label for main input field based on tool"""
	labels = {
		"solve": "Equation (use = or implicit form)",
		"integrate": "Expression to integrate",
		"differentiate": "Expression to differentiate",
		"limit": "Expression for limit",
		"plot": "Function to plot",
		"simplify": "Expression to simplify",
		"matrix": "Matrix (e.g., [[1,2],[3,4]])",
		"latex": "Expression for LaTeX"
	}
	return labels.get(tool_name, "Expression / Equation / Matrix")

def load_example(tool_name):
	"""Load example based on currently selected tool"""
	return EXAMPLES.get(tool_name, "")

def run_tool(tool_name, expr, var, low, up, ordn, s, e, pts, mop, oth):
	inputs = {
		"expression": expr,
		"equation": expr,
		"variable": var or "x",
		"lower": low if low != "" else None,
		"upper": up if up != "" else None,
		"order": int(ordn) if ordn else 1,
		"start": float(s),
		"end": float(e),
		"points": int(pts),
		"matrix_op": mop,
		"matrix": None,
		"other": None,
		"subs": None,
		"dir": "+"
	}
	
	if tool_name == "matrix":
		try:
			inputs["matrix"] = eval(expr, {"__builtins__": {}})
		except Exception:
			return "❌ **Error:** Invalid matrix format. Use [[1,2],[3,4]]", json.dumps({"error": "Invalid matrix input"}, indent=2), None
		if oth:
			try:
				inputs["other"] = eval(oth, {"__builtins__": {}})
			except Exception:
				inputs["other"] = oth
	
	if tool_name == "simplify" and oth:
		try:
			inputs["subs"] = eval(oth, {"__builtins__": {}})
			inputs["evaluate"] = True
		except Exception:
			inputs["subs"] = None
	
	if tool_name == "limit":
		inputs["point"] = oth if oth else "oo"
	
	md, j, img = call_helper(tool_name, inputs)
	return md, j, img

# Gradio UI ---------------------------------------------------------------
demo = gr.Blocks(title="Math MCP Server")

with demo:
	# Header
	gr.Markdown("""
	# 🧮 Math MCP Server
	## Interactive Mathematical Computing Platform
	
	Powered by **SymPy** | **NumPy** | **Matplotlib**
	
	---
	""")
	
	with gr.Row():
		with gr.Column(scale=1, min_width=300):
			gr.Markdown("## 🔧 Tools")
			tool = gr.Radio(
				choices=[
					("📐 Solve Equation", "solve"),
					("∫ Integrate", "integrate"),
					("d/dx Differentiate", "differentiate"),
					("lim Limit", "limit"),
					("📈 Plot Function", "plot"),
					("⚙️ Simplify", "simplify"),
					("🔢 Matrix Operation", "matrix"),
					("LaTeX Output", "latex")
				],
				value="solve",
				label="Select Operation",
				interactive=True
			)
			
			gr.Markdown("---")
			gr.Markdown("### 📚 Quick Start")
			load_btn = gr.Button("📋 Load Example", scale=1, variant="secondary")
			gr.Markdown("""
			**💡 Tip:** Click "Load Example" to populate the input with a sample for the selected tool.
			""")
		
		with gr.Column(scale=2):
			# Tool Info Box
			tool_info = gr.Markdown("### Solve Equation\nFind symbolic and numeric solutions to algebraic equations.\n\n**Example:** `x**2 - 5*x + 6 = 0`")
			
			gr.Markdown("### 📝 Input Parameters")
			
			input_text = gr.Textbox(
				label="Expression / Equation / Matrix",
				lines=3,
				placeholder="Enter your mathematical expression here..."
			)
			
			# Basic parameters (always visible when needed)
			variable = gr.Textbox(
				label="Variable",
				value="x",
				interactive=True,
				visible=True
			)
			
			# Advanced parameters (collapsible)
			with gr.Accordion("⚙️ Advanced Parameters", open=False):
				with gr.Row():
					lower = gr.Textbox(
						label="Lower Bound (use -oo for -∞)",
						value="",
						visible=False
					)
					upper = gr.Textbox(
						label="Upper Bound (use oo for +∞)",
						value="",
						visible=False
					)
				
				with gr.Row():
					order = gr.Number(
						label="Derivative Order",
						value=1,
						precision=0,
						visible=False
					)
					start = gr.Number(
						label="Plot Range Start",
						value=-20,
						visible=False
					)
				
				with gr.Row():
					end = gr.Number(
						label="Plot Range End",
						value=20,
						visible=False
					)
					points = gr.Number(
						label="Resolution (points)",
						value=1000,
						precision=0,
						visible=False
					)
				
				with gr.Row():
					matrix_op = gr.Dropdown(
						["inverse", "determinant", "transpose", "eigen", "multiply"],
						value="inverse",
						label="Matrix Operation",
						visible=False
					)
					other = gr.Textbox(
						label="Other (point, multiplier, subs, etc.)",
						value="",
						visible=False
					)
			
			process_btn = gr.Button("⚡ Compute", variant="primary")
	
	# Output Section
	gr.Markdown("---\n## 📊 Results")
	with gr.Tabs():
		with gr.Tab("📊 JSON Output"):
			output_json = gr.Textbox(
				label="Raw JSON Response",
				lines=25,
				max_lines=40,
				interactive=False
			)
		
		with gr.Tab("📋 Formatted Result"):
			with gr.Row():
				with gr.Column(scale=1):
					output_md = gr.Markdown("_Computation results will appear here..._")
				with gr.Column(scale=1):
					output_img = gr.Image(label="📊 Plot Output", type="pil")
	
	# Footer
	gr.Markdown("""
	---
	### 📖 Documentation & Help
	
	| Operation | Use Case | Example |
	|-----------|----------|---------|
	| **Solve** | Find roots of equations | `x**2 - 4 = 0` → x = ±2 |
	| **Integrate** | Compute integrals | `1/x` → ln(x) |
	| **Differentiate** | Find derivatives | `x**3` → 3x² |
	| **Limit** | Calculate limits | `sin(x)/x` as x→0 → 1 |
	| **Plot** | Visualize functions | Graph any expression |
	| **Simplify** | Reduce expressions | `(x²-1)/(x-1)` → x+1 |
	| **Matrix** | Linear algebra ops | Inverse, determinant, eigenvalues |
	| **LaTeX** | Format for typesetting | Convert to LaTeX syntax |
	
	**Tips:**
	- Use `**` for exponents (e.g., x**2)
	- Use `oo` for infinity
	- Use `-oo` for negative infinity
	- Separate bounds with spaces in the "Other" field when needed
	
	---
	*Math MCP Server v1.0 | Hackathon 2024*
	""")
	
	# Callbacks
	tool.change(
		update_ui,
		[tool],
		[input_text, variable, lower, upper, order, start, end, points, matrix_op, other, tool_info]
	)
	
	load_btn.click(
		load_example,
		[tool],
		[input_text]
	)
	
	process_btn.click(
		run_tool,
		[tool, input_text, variable, lower, upper, order, start, end, points, matrix_op, other],
		[output_md, output_json, output_img]
	)

if __name__ == "__main__":
	demo.launch(server_name="0.0.0.0", server_port=7860)