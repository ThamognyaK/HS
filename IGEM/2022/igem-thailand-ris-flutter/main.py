from os import path
from pathlib import Path
from flask import Flask, render_template
from flask_frozen import Freezer

template_folder = path.abspath('./public')

app = Flask(__name__, template_folder=template_folder)
#app.config['FREEZER_BASE_URL'] = environ.get('CI_PAGES_URL')
app.config['FREEZER_DESTINATION'] = 'public'
app.config['FREEZER_RELATIVE_URLS'] = True
app.config['FREEZER_IGNORE_MIMETYPE_WARNINGS'] = True
freezer = Freezer(app)

@app.cli.command()
def serve():
    freezer.run()

@app.route('/')
def index():
    return render_template('home/index.html')

@app.route('/<page>')
def pages(page):
    return render_template(str((page.lower()) / index.html))

if __name__ == "__main__":
    app.run(port=8080)
