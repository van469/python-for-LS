# this python code answers following questions using titanic.csv file (titanic db link provided in README)


# Q1> How many male and female passengers have missing age data?
import pandas as pd

# load dataset from CSV file
file = pd.read_csv(r'/res/titanic.csv')

file.set_index('PassengerId', inplace=True)

missing_age_male = file[(file['Sex'] =='male') & file['Age'].isnull()]  
missing_age_female= file[(file['Sex'] =='female') & file['Age'].isnull()]

print("missing age males:" + missing_age_male)
print("missing age females:" + missing_age_female)


# Q2> What is the average age of male and female passengers?

avg_male = file[file['Sex'] == 'male']

avg_male = avg_male['Age'].mean()
print("avg. male:" + avg_male)

avg_female = file[file['Sex'] == 'female']

avg_female = avg_female['Age'].mean()
print("avg. female:" + avg_female)


# Q3> Plot a histogram of the age distribution for male and female passengers separately.
import matplotlib.pyplot as plt

#histogram of ages for male
plt.hist(file[file['Sex'] =='male']['Age'], bins=10)
plt.xlabel('Age')
plt.ylabel('Number of males')
plt.title('Age Distribution of Males')
plt.show()

#histogram of ages for female
plt.hist(file[file['Sex'] =='male']['Age'], bins=10)
plt.xlabel('Age')
plt.ylabel('Number of females')
plt.title('Age Distribution of females')
plt.show()
