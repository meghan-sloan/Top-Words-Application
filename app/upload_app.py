import os
from flask import Flask, request, redirect, url_for, flash, send_from_directory, render_template, Response
from werkzeug.utils import secure_filename
from run_nmf_open import *

UPLOAD_FOLDER = '/Users/meghan/top_words_app/Top-Words-Application/app/uploads'
ALLOWED_EXTENSIONS = set(['csv', 'pdf'])

app = Flask(__name__)
app.secret_key = 'verysecretkey'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def change_col_name(col_list):
    changed = [name.lower().strip().replace(' ', '_')for name in col_list]
    return changed

def run_vectorizer(filename, col_list):
    data = pd.read_csv(filename, nrows=41)
    data = data.fillna(value='')
    columns = data.columns
    cols = [col.lower().replace(' ', '_') for col in columns]
    data.columns = cols
    for col in col_list:
        vector, features = preprocess(data, col)
        top_words_df, weights_df = create_dfs(vector, features, data)
        # top_words_df.to_csv('UPLOAD_FOLDER/{}_top_words.csv'.format(col))
        # weights_df.to_csv('UPLOAD_FOLDER/{}_weights.csv'.format(col))
        diversity_overall = overall_summary(vector, features)
        # diversity_overall.to_csv('UPLOAD_FOLDER/results/{}_summary.csv'.format(col))
    return top_words_df, weights_df

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        stop_words = request.form['stop_words']
        col_names = request.form['col_names']
        num_words = request.form['num_words']
        comb_cols = request.form['comb_cols']
        col_list = col_names.split(',')
        changed = change_col_name(col_list)

        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            for col in changed:
                top_words_df, weights_df = run_vectorizer('uploads/{}'.format(filename), changed)
                top_words_df.to_csv('/Users/meghan/top_words_app/Top-Words-Application/app/uploads/results/{}_top_words.csv'.format(col))
                weights_df.to_csv('/Users/meghan/top_words_app/Top-Words-Application/app/uploads/results/{}_weights.csv'.format(col))
            # return '''
            #     <html><body>
            #     <a href="/getTopWords">Top Words CSV.</a>
            #     <a href="/getWeights">Weights CSV.</a>
            #     </body></html>
            #     '''
            return render_template('submit_page.html', ct = weights_df, col = col_names, stop = stop_words)
    return render_template('submit_page.html')

@app.route("/getTopWords")
def getTopWords():
    # with open("outputs/Adjacency.csv") as fp:
    #     csv = fp.read()
    with open('/Users/meghan/top_words_app/Top-Words-Application/app/uploads/results/diversity_means_top_words.csv') as fp:
        csv = fp.read()

    return Response(
        csv,
        mimetype="text/csv",
        headers={"Content-disposition":
                "attachment; filename=top_words.csv"})

@app.route("/getWeights")
def getWeights():
    # with open("outputs/Adjacency.csv") as fp:
    #     csv = fp.read()
    csv = '1,2,3\n4,5,6\n'
    return Response(
        csv,
        mimetype="text/csv",
        headers={"Content-disposition":
                 "attachment; filename=myplot.csv"})

    return render_template('landing_page.html')
#
# @app.route('/results')
# def results():


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5353, debug=True)
