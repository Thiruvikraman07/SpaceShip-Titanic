# SpaceShip-Titanic
Machine Learning Project

This is a competiton dataset from kaggle: https://www.kaggle.com/competitions/spaceship-titanic

Steps Involved:

Handling String Data:

HomePlanet Column has namely 3 different values

Europa-Earth-Mars respectively each were filled with 0-1-2

Similarly

Destination has namely 3 different values

TRAPPIST-1e-PSO J318.5-22-55 Cancri e each were filled with 0-1-2

CryoSleep has bool values which were changed to 0's and 1's

also VIP and Transported has bool values which were changed to 0's and 1's

Similary for handling the Cabin info

first we can see that from data visulization that the first character and last character in the string was significant so made a list made up of those data and found how many different values were there and their values were given proper numerical values.

Handling Missing Data:

The Columns With Missing Datas Were The Following:

1.HomePlanet
2.CryoSleep
3.Cabin
4.Destination
5.Age
6.VIP
7.RoomService
8.FoodCourt
9.ShoppingMall
10.Spa
11.VRDeck
12.Name

we have dropped Name since it doesn't have significant importanace and has missing datas also.

now first we fill the Age with mean of Age value from the column

and when we plot a barplot between VIP and RoomService we can see a correlation between them

<img width="613" alt="Screenshot 2022-12-21 at 8 00 47 PM" src="https://user-images.githubusercontent.com/40484639/208929050-1519e6e3-98a0-4056-80db-cedf9e01e662.png">

that is when people spend more than 300 are VIP and less than 300 are not VIP

so for the missing datas we used this method to fill data

Similarly when we plot VIP and HomePlanet we can see
<img width="580" alt="Screenshot 2022-12-21 at 8 04 02 PM" src="https://user-images.githubusercontent.com/40484639/208929720-7cb382ea-da55-44a3-a432-bbc9c26b0d1a.png">

when VIP == 0 it is planet 2 or else 1

we will drop all other nan values since couldnt further handle it.

Do the same for above for test.csv also

Choose model to predict:

Here I have used a system to select which model will be better for this case and the models I used are:

'xgboost' : xgb.XGBClassifier(),
'lightgbm' : lgb.LGBMClassifier(),
'gradient boosing' : GradientBoostingClassifier(),
'random forest' : RandomForestClassifier(),
'logistic regression': LogisticRegression(),
'naive bayes': GaussianNB(),

we have found that xgboost, lightgbm and gradient boosting were better than others so try these three


