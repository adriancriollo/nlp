from flask import Flask, request, jsonify

app = Flask(__name__)


@app.route("/api/translate", methods=["POST"])
def translate():
    data = request.get_json()
    text = data.get("text", "")
    translated_text = text[::-1]
    return jsonify({"translatedText": translated_text})


if __name__ == "__main__":
    app.run(debug=True)
