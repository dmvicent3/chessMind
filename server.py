from pprint import pprint
from functions import *
from flask import Flask, request, redirect, url_for, jsonify

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            img = np.asarray(bytearray(file.read()))
            board = findBoard(img)
            if board is None:
                return jsonify({'error': 'could not find board'})
            squares = segmentBoard(board)
            prediction = predictImages(squares)
            fen = retFEN(prediction)            
            whiteResult = 'Best white move: ' + calculateBestMove(fen, 'b')
            blackResult = 'Best black move: ' + calculateBestMove(fen, 'w')
            print(whiteResult)
            print(blackResult)
            template = ""
            for count, s in enumerate(squares, 0):
                if count % 8 == 0:
                    template += "<br>"
                base64img = uint8ToBase64(s)
                template += "<img src=" + base64img + \
                    " style='margin: 0px 2px;' width=64 height=64>"
            template += '<h1>' + whiteResult + '</h1>'
            template += '<h1>' + blackResult + '</h1>'
            return '''
            <!doctype html>
            <title>Chess Mind</title>''' + template
        return jsonify({'error': 'invalid image extension'})

    return '''
    <!doctype html>
    <title>Chess Mind</title>
    <h1>Upload board picture</h1>
    <form action="" method=post enctype=multipart/form-data>
      <p><input type=file name=file>
         <input type=submit value=Upload>
    </form>
    '''
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=False)
