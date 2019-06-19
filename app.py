from flask import Flask,render_template,request,redirect
from predict import predict

app = Flask(__name__)


@app.route('/',methods = ['POST','GET'])
def home():
    if request.method == 'GET':
        return render_template('home.html')
    elif request.method == 'POST':
        result = request.form
        print(result)
        if result:
            univs = predict(result)
            return render_template('home.html',result=univs)
        else:
            print('Error')
            return render_template('home.html',error=True)
        

if __name__ == '__main__':
    app.run(debug=True)