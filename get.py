from flask import Flask, render_template, request
app = Flask(__name__)

@app.route('/send', methods=['GET','POST'])
def send():
    if request.method == 'POST':
        features = [None]*32
        for i in range(0,31):
            features[i] = request.form[str(i)]    
        return render_template('result.html')
    else:
        return render_template('index.html')

if __name__ == "__main__":
    app.run()