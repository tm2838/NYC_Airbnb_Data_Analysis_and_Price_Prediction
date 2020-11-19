#!/usr/bin/env python
# coding: utf-8

# In[ ]:


'''
This is for Groundspeed Analytics Take Home Data Challenge 2020. The main task is to design a data structure that
holds both detailed personal information and policy level metrics in a health insurance setting.

Author: Man
Date: 02/09/2020

'''


# In[1]:


import random
import string
from statistics import mean
from dateutil.parser import parse
from datetime import datetime
import numpy as np


# In[2]:


class PolicyHolder():
    '''
    A class for storing policy holder information. 
    Each policy holder is associated with a unique ID and two dictionaries, one for personally identifiable 
    information like name, gender, date of birth, and the other for insured events, containing details like
    LossDate, LossType etc.
    
    '''
    def __init__(self, properties = {'Name':'','Gender':'','DOB':'','SSN':'','SmokingStatus':'','Allergies':'',
                                     'MedicalConditions':''},
                       events = {'LossDate':'','LossType':'','BilledAmount':0,'CoveredAmount':0}):
        '''
        set default values for class attributes and generate a unique ID each time an instance is created
        '''
        self.ID = self.GetUniqueID()
        self.Properties = properties
        self.InsuredEvents = events
        
    def GetUniqueID(self):
        '''
        Inlcuding upper/lower case letters and numbers, so that ID generated could approximate pseudo-random
        
        '''
        return ''.join(random.choices(string.ascii_letters + string.digits, k=15))
    
    def UpdateProperty(self):
        '''
        Prompting user input to update policy holder's information
        
        '''
        Name = input('Please enter policy holder\'s name: ')
        Gender = input('Please enter policy holder\'s gender:  ')
        DOB = input('Please enter policy holder\'s date of birth (mm/dd/yyyy):  ')                
        SSN = input('Please enter policy holder\'s SSN:  ')
        SmokingStatus = input('Please enter policy holder\'s smoking status: ')
        Allergies = input('Please enter any allergy policy holder may have: ')
        MedicalConditions = input('Please enter any medical condition policy holder may have: ')
        self.Properties = {'Name':Name,'Gender':Gender,'DOB':DOB,'SSN':SSN,'SmokingStatus':SmokingStatus,
                           'Allergies':Allergies,'MedicalConditions':MedicalConditions}
        
    def UpdateEvent(self):
        '''
        Prompting user input to update policy holder's claim details
        
        '''
        LossDate = input('Pleae enter policy holder\'s date of incidence (mm/dd/yyyy): ')                
        LossType = input('Pleae enter policy holder\'s type of issue: ')
        BilledAmount = input('Pleae enter policy holder\'s billed amount: ')
        
        while BilledAmount.isdigit() == False:
                print('Invalid input -- {}'.format(BilledAmount))
                BilledAmount = input('Pleae enter policy holder\'s billed amount: ')
                
        CoveredAmount = input('Pleae enter policy holder\'s covered amount: ')
        while CoveredAmount.isdigit() == False:
                print('Invalid input -- {}'.format(CoveredAmount))
                CoveredAmount = input('Pleae enter policy holder\'s covered amount: ')
                
        self.InsuredEvents = {'LossDate':LossDate,'LossType':LossType,'BilledAmount':BilledAmount,
                              'CoveredAmount':CoveredAmount}


# A lot more could be done here. For example, more conditional statements to check the validity of user input, or a 
# prompt to ask user's preference of next step (continue adding policy holders or stop). Due to time limit, I assumed
# only one policy holder is added at each time, and most user inputs are valid.


# In[3]:


class Policy():
    '''
    Policy level class where both information of individual policy holders and aggregate metrics can be accessed.
    
    '''
    
    def __init__(self,PolicyHolders = [],Metrics = {'totalCovered':0,'claimsPerYr':{},'averageAge':0}):
        self.PolicyHolders = PolicyHolders
        self.Metrics = Metrics
        
    def Add_PolicyHolder(self):
        '''
        Adding new policy holder's information by instantiating PolicyHolder class and prompting corresponding user 
        input. At the same time, aggregate metrics would be automatically updated.
        
        '''
        new_holder = PolicyHolder()
        new_holder.UpdateProperty()
        print('New policy holder: {}'.format(new_holder.Properties['Name']) + '\n'
              +'New policy holder\'s ID : {}'.format(new_holder.ID))
        self.PolicyHolders.append(new_holder)   
        self.UpdateMetrics()
                 
    def Add_InsuredEvent(self):
        '''
        Adding new events associated with a specific policy holder by prompting corresponding user 
        input. At the same time, aggregate metrics would be automatically updated.
        
        '''
        UserID = str(input('Please enter the ID of the individual: '))
        for py_holder in self.PolicyHolders:
            if py_holder.ID == UserID:
                py_holder.UpdateEvent()
        self.UpdateMetrics()
        
    def ListInsureds(self):
        '''
        Listing ID and relavant personal identifiable information of all policy holders
        '''
        for py_holder in self.PolicyHolders:
            print('ID'  + ': '+py_holder.ID)
            for key, value in py_holder.Properties.items():
                print(key,":",value)
            print('\n')
            
    def ListEvents(self):
        '''
        Listing all insured events of a policy holder with user input ID
        '''
        UserID = str(input('Please enter the ID of the individual: '))
        for py_holder in self.PolicyHolders:
            if py_holder.ID == UserID:
                print('ID' + ': '+py_holder.ID)
                for key, value in py_holder.InsuredEvents.items():
                    print(key,":",value)
                 
    def UpdateMetrics(self):
        '''
        Append covered amount/age/loss year into a list respectively and calculate the aggregate metric.
        
        '''
        all_covered = [float(py_holder.InsuredEvents['CoveredAmount']) for py_holder in self.PolicyHolders]
        self.Metrics['totalCovered'] = sum(all_covered)
        
        all_age = [(datetime.today().year - parse(py_holder.Properties['DOB']).year) 
                   for py_holder in self.PolicyHolders
                   if py_holder.Properties['DOB'] != ''] 
        if all_age != []:
            self.Metrics['averageAge'] = mean(all_age)
       
        all_yr = np.array([parse(py_holder.InsuredEvents['LossDate']).year
                 for py_holder in self.PolicyHolders
                 if py_holder.InsuredEvents['LossDate'] != ''])
        
        (unique_yr, counts) = np.unique(all_yr, return_counts=True)
        for i in range(len(unique_yr)):
            self.Metrics['claimsPerYr'].update({unique_yr[i]: counts[i]})


# In[ ]:


'''
Privacy is an important issue to take into account when dealing with personal identifiable information. 
Maybe after storing pII such as SSN, a mask could be given to it, so that an authorized user could acess it 
but an unauthorized user could only get the blurred version.

'''

