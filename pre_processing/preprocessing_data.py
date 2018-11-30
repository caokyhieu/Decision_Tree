import numpy as np
import pandas as pd

### Discrete for age
def categorical_age(age):
	if age<16:
		return '<16'
	elif age<=50:
		return '16-50'
	else:
		return '>50'

# discrete for fare	
def categorical_fare(fare):
	if fare < 20:
		return '<20'
	elif fare < 50:
		return '20-50'
	else:
		return '>50'

### From name we can attract the personal status for each person
def personal_status(name):
	if ('Master.' in name):
		return 'Master'
	elif 'Dr.' in name:
		return 'Doctor'
	elif ('Rev.' in name) or ('Col.' in name):
		return 'Religion'
	elif('Capt' in name) or ('Major.' in name):
		return 'Millitary'
	elif 'Mrs' in name:
		return 'Mrs'
	elif ('Miss' in name) or ('Mlle.' in name):
		return 'Miss'
	else:
		return 'Normal'



def preprocessing_titanic(train="data/train.csv",test='data/test.csv'):
	

	train_df = pd.read_csv(train)
	test_df = pd.read_csv(test)

	# Fill missing data for age in train and test
	train_df[['Age']] = train_df[['Age']].fillna(train_df['Age'].mean())
	test_df[['Age']] = test_df[['Age']].fillna(train_df['Age'].mean())

	#
	test_df[['Fare']] = test_df[['Fare']].fillna(train_df['Fare'].mean())

	train_df[['Embarked']] = train_df[['Embarked']].fillna(train_df['Embarked'].mode()[0])

	# we just take the type of cabin depend on first letter of this attribute

	train_df[['Cabin']] = train_df[['Cabin']].fillna('Undifined')
	test_df[['Cabin']] = test_df[['Cabin']].fillna('Undifined')
	train_df[['Cabin']] = train_df[['Cabin']].apply(lambda x : x.str[0])
	test_df[['Cabin']] = test_df[['Cabin']].apply(lambda x: x.str[0])

	### from chart, I choose 3 is the threshold for SibSp, 4 for Parch
	train_df['SibSp'] = train_df['SibSp'].apply(lambda x: '<=2' if x<=3 else '>2')
	train_df['Parch'] = train_df['Parch'].apply(lambda x: '<=3' if x<=3 else '>3')


	test_df['SibSp'] = test_df['SibSp'].apply(lambda x: '<=2' if x<=3 else '>2')
	test_df['Parch'] = test_df['Parch'].apply(lambda x: '<=3' if x<=3 else '>3')

	## we can divide age and Fare to 3 intervals 
	train_df['Age'] = train_df['Age'].apply(categorical_age)
	train_df['Fare'] = train_df['Fare'].apply(categorical_fare)

	test_df['Age'] = test_df['Age'].apply(categorical_age)
	test_df['Fare'] = test_df['Fare'].apply(categorical_fare)

	## personal_status
	train_df['personal_status'] = train_df['Name'].apply(personal_status)
	test_df['personal_status'] = test_df['Name'].apply(personal_status)

	## Drop the unused attributes
## We just use ['Sex','Pclass','Parch','SibSp','Age','Fare','Cabin','Embarked','personal_status']
	train_df = train_df.drop(['PassengerId','Name','Ticket'],axis=1)
	test_df = test_df.drop(['PassengerId','Name','Ticket'],axis=1)


	return (train_df,test_df)



		