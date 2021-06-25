from flask import Flask,render_template,request
app=Flask(__name__)
import pickle

#open a file,where you stored the pickle data
file=open('model.pkl','rb')
clf=pickle.load(file)
file.close()

@app.route('/', methods=["GET", "POST"])
def hello_world():
    if request.method == "POST":
        myDict=request.form
        Fever=int(myDict['Fever'])
        Age=int(myDict['Age'])
        BodyPain=int(myDict['BodyPain'])
        RunnyNose=int(myDict['RunnyNose'])
        DiffBreath=int(myDict['DiffBreath'])
        #Code for inference
        inputFeature=[Fever,BodyPain,Age,RunnyNose,DiffBreath]
        infProb=clf.predict_proba([inputFeature])[0][1]
        print(infProb)
        return render_template('show.html',inf=round(infProb*100))
    #return 'Hello! world' + str(infProb)
    return render_template('index.html')


if __name__ == "__main__":
    app.run(debug=True)
