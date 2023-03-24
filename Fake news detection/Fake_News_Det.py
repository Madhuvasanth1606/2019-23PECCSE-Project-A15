from flask import Flask, render_template, request
from lightgbm import LGBMClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split

app = Flask(__name__)
tfvect = TfidfVectorizer(stop_words='english', max_df=0.7)
loaded_model = pickle.load(open('model.pkl', 'rb'))
dataframe = pd.read_csv('news.csv',low_memory=False)
dataframe = dataframe.fillna(' ')


x = dataframe['text']
y = dataframe['label']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

def fake_news_det(news):
    tfid_x_train = tfvect.fit_transform(x_train)
    tfid_x_test = tfvect.transform(x_test)
    input_data = [news]
    vectorized_input_data = tfvect.transform(input_data)
    prediction = loaded_model.predict(vectorized_input_data)
    return prediction

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        pred = fake_news_det(message)
        print(pred)

        politics="political politician law government governance diplomatic polity public administration diplomatical political opportunism partisanship election republic politics aristotle politically politicalscience localgovernment partisan politics economics state debate ideologies monarchy politic democratic religion politicking policy nationalism smooth  wars  political system activism  politicians  suave  politicos  morality  politicizing   regionalism  governor  judiciary  policymaking  political expediency  expedient  geopolitics  parliament history  society police  republicanism  policies  sagacious  demagoguery  elections  voting  electoral  populist  ban  conspiracy polis  principle  politics  sociopolitical social  mudslinging  politicdrama  divisive  divisiveness  factionalism  ideological  statecraft  journalism  parochialism   cynicism  gubernatorial  ethics  worldpopulism  regime  economy  power  partisan bickering  pandering rhetoric  bureaucracy  intrigue  federal government  personal democracy identity tribes  controversial  civil  legal  unitednations  lawmaking  allegiance  justice  celibacy  governmental  civilised  sinfulness  stuffy  meddles  unclothed crystallizing  psyche  policy-making  anarchy  engrained  latinisation  pols  humorless  clubby  metabolizes  political parties politicize  multinational  realpolitik  crist  demagogue  supremacy  nations  politick  statesmanship  conservatism  pragmatism  electioneering  electorate  idealism  dukes  sovereign state slinging  tribalism  partisan  curfew government indiangovernmrnt donald trump obama narendra modi Modi minister exminister parliment council barack department"
        
        weather="Weather weather rain cloud rainbow temperature pressure overcast shower sunrise dry  tornado  sunset  humidity  cold  heat  wind cloudy  heat wave  fog  breeze  humid  lightning  blustery  humidity  thunder  snow  heat index  thunderstorm  downpour  drought  tropical  water cycle  temperate  moisture drizzle  warm  hail  icicle  climate  storm  flood  muggy  gale  flash flood  atmosphere  cold front  mist  isobar  cold snap  condensation  forecast  ice storm  freeze barometric  gust snowfall rainfall raining sunny monsoon"

        sports="aerobics archer archery arena arrow athlete athletics axel badminton ball base baseball  basketball bat baton batter batting biathlon bicycle bicycling bike biking billiards bobsleigh bocce boomerang boules bow bowler bowling boxer boxing bronze medal bunt cricket shuttle football goal billiards wicket bowler boldout catchout losethematch cricket play cricketer"

        politics=politics.split()
        message=message.split()
        weather=weather.split()
        sports=sports.split()

        politics=tuple(politics)
        message=tuple(message)
        weather=tuple(weather)
        sports=tuple(sports)

        for x in range(len(message)):
            if(message[x] in politics):
                a="Politics"
            elif(message[x] in weather):
                a="weather"
            elif(message[x] in sports):
                a="sports"
            else:
                a="Others"

        return render_template('index.html', prediction=pred,newstype=a)
    else:
        return render_template('index.html', prediction="Something went wrong")
    


   


    
if __name__ == '__main__':
   app.run(host='0.0.0.0',port='420',debug=True)
