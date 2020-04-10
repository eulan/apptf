from flask import Flask, render_template, request
import test


app = Flask(__name__)

@app.route('/', methods = ["POST", "GET"])
def home():
	x = ""
	result = ""
	y=""
	if request.method == "POST":
		x = request.form["string"]
		y = request.form["NumChar"]
		if not y:
			y = 1000
		else:
			y = int(y)
		m = test.ModelGenerator(str(x))
		m.transformer()
		result = test.generate_text(m.model, start_string=m.text, num_generate = y)
	return render_template("index.html", content = result)

if __name__=="__main__":
	app.run(debug = True)
