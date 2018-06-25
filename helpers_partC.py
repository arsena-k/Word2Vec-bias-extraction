# -*- coding: utf-8 -*-
"""
Created on Sun Jun 24 16:19:04 2018

@author: Alina Arseniev & Jacob Foster

Helper functions for Part C
"""
#load up (and install if needed) libraries
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold 
from sklearn import preprocessing
from statistics import mode, mean, stdev


def normalizeME(vec):
    b= vec.reshape(-1,1)
    c=b/np.linalg.norm(b, ord=2) 
    #print np.linalg.norm(1)
    return c
    
def project(A,B): #projection of A onto B
    numerator= A.dot(B) #should be a scalar
    denominator= np.linalg.norm(B,1) #should be another scalar
    scalarproject = numerator / denominator #divide numerator by denominator, this gives you the scalar projection (LENGTH aka MAGNITUDE) of b onto a
    return scalarproject
    #if want to get the vector projection (i.e. magnutide and DIRECTION),  result 1 (MAGNITUDE) by vector a, and divded by (norm of A)^2
    
    
def select_training_set(trainingset, yourmodelhere): #options are: gender, moral, health, ses
    #gender is the training set used in corresponding paper
    #gender_2 has fewer precise gender words like "he" vs "she" than set 1,  and some more noise via words that are gendered but less clearcut than Set1)
    #gender_3 even fewer precise gender words like "he" vs "she" than set 1,  and same added noise as training set 2

    if trainingset=='gender_2':
        pos_word_list=[ 'girl', 'girls', 'her', 'hers', 'herself', 'she', 
            'lady', 'gal', 'gals', 'madame', 'ladies', 'lady',
          'mother', 'mothers', 'mom', 'moms', 'mommy', 'mama', 'ma', 'granddaughter', 'daughter', 'daughters', 'aunt', 'godmother', 
          'grandma', 'grandmothers', 'grandmother', 'sister', 'sisters', 'aunts', 'stepmother', 'granddaughters', 'niece',
        'fiancee', 'ex_girlfriend', 'girlfriends', 'wife', 'wives', 'girlfriend', 'bride', 'brides', 'widow',
           'twin_sister', 'younger_sister', 'teenage_girl', 'teenage_girls', 'eldest_daughter','estranged_wife', 'schoolgirl',
        'businesswoman', 'congresswoman' , 'chairwoman', 'councilwoman', 'waitress', 'hostess', 'convent', 'heiress', 
           'saleswoman', 'queen', 'queens', 'princess', 'nun' , 'nuns', 'heroine', 'actress', 'actresses', 'uterus', 'vagina', 'ovarian_cancer',
        'maternal', 'maternity', 'motherhood', 'sisterhood', 'girlhood', 'matriarch', 'sorority', 'mare', 'hen', 'hens', 'filly', 'fillies',
          'deer', 'older_sister', 'oldest_daughter', 'stepdaughter', 'pink',  'cute', 'dependent', 'nurturing', 'hysterical', 'bitch',  'dance', 'dancing'] 
        neg_word_list=['boy', 'boys', 'him', 'his', 'himself', 'he', 'guy', 'dude',
            'dudes', 'sir', 'guys', 'gentleman','father', 'fathers', 'dad', 'dads', 'daddy', 'papa', 'pa', 'grandson' , 'son', 'sons', 'uncle', 'godfather', 
        'grandpa', 'grandfathers', 'grandfather', 'brother', 'brothers' , 'uncles', 'stepfather', 'grandsons', 'nephew',
           'fiance', 'ex_boyfriend', 'boyfriends', 'husband', 'husbands', 'boyfriend', 'groom', 'grooms', 'widower',
            'twin_brother', 'younger_brother', 'teenage_boy', 'teenage_boys', 'eldest_son', 'estranged_husband', 'schoolboy',
            'businessman', 'congressman', 'chairman', 'councilman', 'waiter', 'host', 'monastery', 'heir', 'salesman', 
            'king', 'kings', 'prince', 'monk', 'monks', 'hero', 'actor', 'actors', 'prostate', 'penis', 'prostate_cancer', 
        'paternal', 'paternity', 'fatherhood', 'brotherhood', 'boyhood', 'patriarch', 'fraternity', 'stallion', 'rooster', 'roosters', 'colt',
           'colts', 'bull', 'older_brother', 'oldest_son', 'stepson', 'blue' ,'manly', 'independent', 'aggressive', 'angry', 'jerk', 'wrestle', 'wrestling'  ]
        pos_word_replacement='woman'
        neg_word_replacement='man'
    elif trainingset=='gender_3':
        pos_word_list=['madame', 'ladies', 'lady',
          'mother', 'mothers', 'mom', 'mama', 'granddaughter', 'daughter', 'daughters', 'aunt', 'godmother', 
          'grandma', 'grandmothers', 'grandmother', 'sister', 'sisters', 'aunts', 'stepmother', 'granddaughters', 'niece',
        'fiancee', 'ex_girlfriend', 'girlfriends', 'wife', 'wives', 'girlfriend', 'bride', 'brides', 'widow',
           'twin_sister', 'younger_sister', 'teenage_girl', 'teenage_girls', 'eldest_daughter','estranged_wife', 'schoolgirl',
        'businesswoman', 'congresswoman' , 'chairwoman', 'councilwoman', 'waitress', 'hostess', 'convent', 'heiress', 
           'saleswoman', 'queen', 'queens', 'princess', 'nun' , 'nuns', 'heroine', 'actress', 'actresses', 'uterus', 'vagina', 'ovarian_cancer',
        'maternal', 'maternity', 'motherhood', 'sisterhood', 'girlhood', 'matriarch', 'sorority', 'mare', 'hen', 'hens', 'filly', 'fillies',
          'deer', 'older_sister', 'oldest_daughter', 'stepdaughter', 'pink', 'cute', 'dependent', 'nurturing', 'hysterical', 'bitch',  'dance', 'dancing']
        neg_word_list=['sir', 'guys', 'gentleman','father', 'fathers', 'dad', 'papa', 'grandson' , 'son', 'sons', 'uncle', 'godfather', 
        'grandpa', 'grandfathers', 'grandfather', 'brother', 'brothers' , 'uncles', 'stepfather', 'grandsons', 'nephew',
           'fiance', 'ex_boyfriend', 'boyfriends', 'husband', 'husbands', 'boyfriend', 'groom', 'grooms', 'widower',
            'twin_brother', 'younger_brother', 'teenage_boy', 'teenage_boys', 'eldest_son', 'estranged_husband', 'schoolboy',
            'businessman', 'congressman', 'chairman', 'councilman', 'waiter', 'host', 'monastery', 'heir', 'salesman', 
            'king', 'kings', 'prince', 'monk', 'monks', 'hero', 'actor', 'actors', 'prostate', 'penis', 'prostate_cancer', 
        'paternal', 'paternity', 'fatherhood', 'brotherhood', 'boyhood', 'patriarch', 'fraternity', 'stallion', 'rooster', 'roosters', 'colt',
           'colts', 'bull', 'older_brother', 'oldest_son', 'stepson', 'blue' ,'manly', 'independent', 'aggressive', 'angry', 'jerk', 'wrestle', 'wrestling'  ]
        pos_word_replacement='woman'
        neg_word_replacement='man'
    elif trainingset=='gender':
        pos_word_list=['womanly', 'my_wife', 'my_mom', 'my_grandmother', 'woman', 'women', 'girl', 'girls', 'her', 'hers', 'herself', 'she', 
            'lady', 'gal', 'gals', 'madame', 'ladies', 'lady',
          'mother', 'mothers', 'mom', 'moms', 'mommy', 'mama', 'ma', 'granddaughter', 'daughter', 'daughters', 'aunt', 'godmother', 
          'grandma', 'grandmothers', 'grandmother', 'sister', 'sisters', 'aunts', 'stepmother', 'granddaughters', 'niece',
          'fiancee', 'ex_girlfriend', 'girlfriends', 'wife', 'wives', 'girlfriend', 'bride', 'brides', 'widow',
           'twin_sister', 'younger_sister', 'teenage_girl', 'teenage_girls', 'eldest_daughter','estranged_wife', 'schoolgirl',
          'businesswoman', 'congresswoman' , 'chairwoman', 'councilwoman', 'waitress', 'hostess', 'convent', 'heiress', 
           'saleswoman', 'queen', 'queens', 'princess', 'nun' , 'nuns', 'heroine', 'actress', 'actresses', 'uterus', 'vagina', 'ovarian_cancer',
           'maternal', 'maternity', 'motherhood', 'sisterhood', 'girlhood', 'matriarch', 'sorority', 
         'older_sister', 'oldest_daughter', 'stepdaughter']
        neg_word_list=['manly', 'my_husband', 'my_dad','my_grandfather', 'man', 'men', 'boy', 'boys', 'him', 'his', 'himself', 'he', 'guy', 'dude',
            'dudes', 'sir', 'guys', 'gentleman','father', 'fathers', 'dad', 'dads', 'daddy', 'papa', 'pa', 'grandson' , 'son', 'sons', 'uncle', 'godfather', 
           'grandpa', 'grandfathers', 'grandfather', 'brother', 'brothers' , 'uncles', 'stepfather', 'grandsons', 'nephew',
           'fiance', 'ex_boyfriend', 'boyfriends', 'husband', 'husbands', 'boyfriend', 'groom', 'grooms', 'widower',
            'twin_brother', 'younger_brother', 'teenage_boy', 'teenage_boys', 'eldest_son', 'estranged_husband', 'schoolboy',
            'businessman', 'congressman', 'chairman', 'councilman', 'waiter', 'host', 'monastery', 'heir', 'salesman', 
            'king', 'kings', 'prince', 'monk', 'monks', 'hero', 'actor', 'actors', 'prostate', 'penis', 'prostate_cancer', 
           'paternal', 'paternity', 'fatherhood', 'brotherhood', 'boyhood', 'patriarch', 'fraternity', 
           'older_brother', 'oldest_son', 'stepson']
        pos_word_replacement='woman'
        neg_word_replacement='man'
    elif trainingset=='moral':
        pos_word_list= ['good', 'benevolent', 'nice', 'caring', 'conscientious', 'polite', 'fair', 'virtue', 'respect', 'responsible', 
            'selfless', 'unselfish', 'sincere', 'truthful', 'wonderful', 'justice', 'innocent', 'innocence',
           'complement', 'sympathetic', 'virtue', 'right', 'proud', 'pride','respectful', 'appropriate', 'pleasing', 'pleasant', 
            'pure', 'decent', 'pleasant', 'compassion' , 'compassionate', 'constructive','graceful', 'gentle', 'reliable',
           'careful', 'help', 'decent' , 'moral', 'hero', 'heroic', 'heroism', 'honest', 'honesty',
           'selfless', 'humility', 'humble', 'generous', 'generosity', 'faithful', 'fidelity', 'worthy', 'tolerant',
            'obedient', 'pious', 'saintly', 'angelic', 'virginal', 'sacred', 'reverent', 'god', 'hero', 'heroic', 
            'forgiving', 'saintly','holy', 'chastity', 'grateful', 'considerate', 'humane', 
            'trustworthy', 'loyal', 'loyalty', 'empathetic', 'empathy', 'clean', 'straightforward', 'pure']
        neg_word_list= ['bad', 'evil', 'mean', 'uncaring', 'lazy', 'rude', 'unfair', 'sin', 'disrespect','irresponsible', 
           'self_centered', 'selfish', 'insincere', 'lying', 'horrible', 'injustice', 'guilty', 'guilt', 
            'insult', 'unsympathetic', 'vice', 'wrong', 'ashamed', 'shame', 'disrespectful', 'inappropriate', 'vulgar', 'crude', 
            'dirty', 'obscene', 'offensive', 'cruelty','brutal', 'destructive', 'rude', 'harsh', 'unreliable',
            'careless', 'harm', 'indecent', 'immoral', 'coward', 'cowardly', 'cowardice', 'dishonest', 'dishonesty',
            'narcissistic', 'arrogance', 'arrogant', 'greedy', 'greed', 'betray', 'betrayal', 'unworthy', 'intolerant', 
             'defiant', 'rebellious', 'demonic','devilish', 'promiscuous', 'profane', 'irreverent', 'devil', 'villain', 'villainous', 
            'vindictive', 'diabolical', 'unholy', 'promiscuity', 'ungrateful', 'thoughtless', 'inhumane',
            'untrustworthy', 'treacherous', 'treachery', 'callous', 'indifference', 'dirty', 'manipulative', 'impure' ]
        pos_word_replacement='moral'
        neg_word_replacement='immoral'
    elif trainingset=='health':
        pos_word_list= ['fertile', 'help_prevent', 'considered_safe', 'safer', 'healthy', 'healthy', 'healthy', 'healthy', 'healthy',
            'healthful', 'well_balanced', 'natural', 'healthy', 'athletic','physically_active', 'health',
            'health', 'nutritious','nourishing', 'stronger', 'strong','wellness', 'safe', 'nutritious_food','exercise',
            'physically_fit', 'unprocessed', 'healthier_foods', 'nutritious_foods', 'nutritious', 'nutritious',
           'healthy_eating', 'healthy_diet', 'healthy_diet', 'nourishing', 'nourished', 'regular_exercise', 'safety', 'safe', 
            'helpful', 'beneficial', 'healthy', 'healthy', 'sturdy', 'lower_risk', 'reduced_risk', 'decreased_risk', 'nutritious_foods', 'whole_grains', 'healthier_foods',
            'healthier_foods', 'physically_active', 'physical_activity', 'nourished', 'vitality', 'energetic', 'able_bodied',
            'resilience', 'strength', 'less_prone', 'sanitary', 'clean',  'healing', 'heal', 'salubrious']   
        neg_word_list= ['infertile', 'cause_harm','potentially_harmful','riskier', 'unhealthy', 'sick', 'ill', 'frail', 'sickly', 
            'unhealthful','unbalanced', 'unnatural', 'dangerous', 'sedentary', 'inactive', 'illness', 
            'sickness', 'toxic', 'unhealthy', 'weaker', 'weak', 'illness', 'unsafe', 'unhealthy_foods', 'sedentary',
            'inactive', 'highly_processed', 'processed_foods', 'junk_foods', 'unhealthy_foods', 'junk_foods',
               'processed_foods', 'processed_foods', 'fast_food', 'unhealthy_foods', 'deficient', 'sedentary', 'hazard','hazardous', 
            'harmful', 'injurious',  'chronically_ill', 'seriously_ill', 'frail', 'higher_risk', 'greater_risk', 'increased_risk', 'fried_foods', 'fried_foods',
            'fatty_foods', 'sugary_foods', 'sedentary', 'physical_inactivity', 'malnourished', 'lethargy', 'lethargic', 'disabled',
            'susceptibility', 'weakness', 'more_susceptible', 'filthy', 'dirty', 'harming', 'hurt', 'deleterious']
        pos_word_replacement='healthy'
        neg_word_replacement='ill'
    elif trainingset=='ses':
        pos_word_list=['wealth', 'wealthier', 'wealthiest', 'affluence', 'prosperity', 'wealthy', 'affluent', 'affluent', 'prosperous',
                'prosperous','prosperous','disposable_income',  'wealthy','suburban','luxurious','upscale','upscale', 'luxury', 
                'richest', 'privileged', 'moneyed', 'privileged', 'privileged', 'educated', 'employed', 
                'elite', 'upper_income', 'upper_class', 'employment', 'riches', 'millionaire', 'aristocrat', 'college_educated',
                'abundant', 'abundance', 'luxury', 'profitable', 'profit', 'well_educated', 'elites', 'heir', 'well_heeled', 
                'white_collar', 'higher_incomes', 'bourgeois', 'fortunate', 'successful','economic_growth', 'prosper', 'suburbanites']
        neg_word_list= ['poverty', 'poorer', 'poorest', 'poverty', 'poverty', 'impoverished', 'impoverished',  'needy',  'impoverished',
                 'poor', 'needy', 'broke', 'needy', 'slum', 'ghetto', 'slums', 'ghettos', 'poor_neighborhoods', 
                'poorest', 'underserved', 'disadvantaged','marginalized', 'underprivileged', 'uneducated', 'unemployed', 
                'marginalized', 'low_income', 'underclass','unemployment', 'rags', 'homeless', 'peasant', 'college_dropout', 
                'lacking', 'lack', 'squalor', 'bankrupt', 'debt', 'illiterate' ,'underclass', 'orphan',  'destitute', 
                'blue_collar', 'low_income', 'neediest', 'less_fortunate', 'unsuccessful', 'economic_crisis', 'low_wage', 'homeless']
        pos_word_replacement='wealthy'
        neg_word_replacement='poor'
    pos_words=[]
    neg_words=[]
    pos_word_list_checked=[]
    neg_word_list_checked=[]
    for i in pos_word_list:
        try:
            pos_words.append(yourmodelhere[i])
            pos_word_list_checked.append(i)
        except KeyError:
            #print(str(i) +  ' was not in this Word2Vec models vocab, and has been replaced with: ' + str(pos_word_replacement) ) #uncomment this to be alerted each time a training word doesn't exist in your model vocab
            pos_words.append(yourmodelhere[pos_word_replacement])
            pos_word_list_checked.append(pos_word_replacement)
    for i in neg_word_list:
        try:
            neg_words.append(yourmodelhere[i])
            neg_word_list_checked.append(i)
        except KeyError:
            #print(str(i) +  ' was not in this Word2Vec models vocab, and has been replaced with: ' + str(neg_word_replacement) ) #uncomment this to be alerted each time a training word doesn't exist in your model vocab
            neg_words.append(yourmodelhere[neg_word_replacement])
            neg_word_list_checked.append(neg_word_replacement)

    print("Number of pos words: " + str(len(pos_words)) + " Number of neg words: " + str(len(neg_words)))
    train_classes_pos=np.array(np.repeat(1, len(pos_words)).tolist()) #1 is feminine/moral/healthy/rich by default, we will need to find the correct labels inductively
    train_classes_neg=np.array(np.repeat(0, len(neg_words)).tolist()) #0 is masculine/immoral/unhealthy/poor by default, we will need to find the correct labels inductively
    
    pos_words=np.asarray(pos_words)
    neg_words= np.asarray(neg_words)
    return(pos_word_list_checked, neg_word_list_checked, pos_words, neg_words, train_classes_pos, train_classes_neg)
    
#get mean of a set of vectors
def MEAN_get_directionVec_unipolar(words, yourmodelhere, train_index_list=None):
    if train_index_list is not None:
        Ndiff=len(words[train_index_list])
        biggie2= np.reshape(words[train_index_list], (Ndiff,len(yourmodelhere['word']))) #now a Ndiff by #dimensions, menasured by len(yourmodelhere['word']), matrix, can check with print(biggie2.shape)
    else:
        Ndiff= len(words) 
        biggie2= np.reshape(words, (Ndiff,len(yourmodelhere['word']))) #now a Ndiff by #dimensions, menasured by len(yourmodelhere['word']), matrix, can check with print(biggie2.shape)
    biggie2= preprocessing.normalize(biggie2, norm='l2')

    all_vecs_mat= [] #this will be used to make the covariance matrix, cov_mat
    for i in range(0,Ndiff):
        all_vecs_mat.append(biggie2[i,:])
        
    direction = np.mean(all_vecs_mat,0)
    assert direction.shape == (len(yourmodelhere['word']),)
    extracteddirectionVec= np.hstack(normalizeME(direction)) #ensure normalized 
    return(extracteddirectionVec)
    
#get difference between two sets vectors (using MEAN_get_directionVec_unipolar)
def MEAN_get_directionVec_differences(pos_words_, neg_words_, yourmodelhere, train_index_list=None):
    return(MEAN_get_directionVec_unipolar(pos_words_, yourmodelhere, train_index_list)- MEAN_get_directionVec_unipolar(neg_words_, yourmodelhere, train_index_list))    

    
#a function to select a testing set of words
def select_testing_set(testingset, yourmodelhere):
    if testingset=='gender':
        test_word_list= ['goddess', 'single_mother', 'girlish', 'feminine', 'young_woman', 'little_girl', 'ladylike', 'my_mother', 
           'teenage_daughter', 'mistress', 'great_grandmother', 'adopted_daughter', 'femininity', 'motherly', 'matronly', 
           'showgirl', 'housewife', 'vice_chairwoman', 'co_chairwoman', 'spokeswoman', 'governess', 'divorcee', 'spinster', 
           'maid', 'countess', 'pregnant_woman', 'landlady', 'seamstress', 'young_girl', 'waif', 'femme_fatale','comedienne',
            'boyish', 'masculine',  'lad', 'policeman', 'macho', 'gentlemanly', 'machismo',  'teenage_son', 
            'beau', 'great_grandfather', 'tough_guy', 'masculinity', 'bad_boy', 'spokesman', 'baron', 'adult_male', 'landlord', 'fireman', 'mailman', 'vice_chairman', 
           'co_chairman','young_man', 'bearded', 'mustachioed', 'con_man', 'homeless_man', 'gent', 'strongman']
        test_classes=np.repeat(1, 32).tolist() #1 is feminine
        masc2=np.repeat(0, 28).tolist() #0 is masculine
        for i in masc2:
            test_classes.append(i) 
    elif testingset=='moral':
        test_word_list= ['great', 'best', 'faith', 'chaste', 'wholesome', 'noble', 'honorable', 'immaculate', 'gracious', 
           'courteous', 'delightful', 'earnest', 'amiable', 'admirable', 'disciplined', 'patience', 'integrity',
            'restraint', 'upstanding', 'diligent', 'dutiful', 'loving', 'righteous','respectable', 'praise', 'devout', 'forthright',
            'depraved', 'repulsive', 'repugnant', 'corruption', 'vicious', 'unlawful', 'outrage',  'shameless', 'perverted',
            'filthy', 'lewd', 'subversive', 'sinister', 'murderous', 'perverse', 
           'monstrous', 'homicidal', 'indignant', 'misdemeanor', 'degenerate', 'malevolent', 'illegal','terrorist','terrorism',  
             'cheated', 'vengeful', 'culpable','vile', 'hateful', 'abuse', 'abusive', 'criminal', 'deviant']
        test_classes=np.repeat(1, 27 ).tolist() #1 is feminine
        masc2=np.repeat(0,33).tolist() #0 is masculine
        for i in masc2:
            test_classes.append(i)
    elif testingset=='health':
        test_word_list= [ 'balanced_diet', 'healthfulness', 'fiber', 'jogging', 'stopping_smoking', 'vigor', 
          'active', 'fit', 'flourishing', 'sustaining', 'hygienic', 'hearty', 'enduring', 'energized', 'wholesome', 
           'holistic', 'healed', 'fitter', 'health_conscious', 'more_nutritious', 'live_longer',  'exercising_regularly',
           'healthier_choices', 'healthy_habits', 'healthy_lifestyle', 'healthful_eating', 'immune', 
            'deadly', 'diseased',  'adverse', 'risky', 'fatal', 'filthy', 'epidemic', 'crippling', 'carcinogenic', 'carcinogen',
           'crippled', 'afflicted', 'contaminated', 'fatigued', 'detrimental', 'bedridden', 'incurable', 'hospitalized',
           'infected', 'ailing', 'debilitated', 'poisons', 'disabling', 'life_threatening', 'debilitating', 
           'chronic_illness', 'artery_clogging', 'hypertension','disease', 'stroke',
            'plague', 'poisonous', 'smoking']
        test_classes=np.repeat(1, 27).tolist() #1 is feminine
        masc2=np.repeat(0, 33 ).tolist() #0 is masculine
        for i in masc2:
            test_classes.append(i) 
    elif testingset=='ses':
        test_word_list= ['rich', 'billionaire', 'banker',  'fortune', 'heiress', 'cosmopolitan', 'ornate', 'entrepreneur', 'sophisticated',
                'aristocratic', 'investor', 'highly_educated', 'better_educated',  'splendor', 
               'businessman', 'opulent', 'multimillionaire', 'philanthropist', 'estate', 'estates', 'chateau', 'fortunes', 
               'financier', 'young_professionals','tycoon', 'baron', 'grandeur', 'magnate', 
               'investment_banker', 'venture_capitalist', 'upwardly_mobile', 'highly_skilled', 'yuppies', 'genteel',
                         'homelessness', 'ruin', 'ruined', 'downtrodden', 'less_affluent',
                'housing_project', 'homeless_shelters', 'indigent', 'jobless', 'welfare',  
                'temporary_shelters','housing_projects', 'subsidized_housing', 'starving', 'beggars', 'orphanages',
                'dispossessed', 'uninsured', 'welfare_recipients', 'food_stamps', 
                'malnutrition',  'underemployed', 'disenfranchised', 'servants', 'displaced', 'poor_families'] 
        test_classes=np.repeat(1, 34).tolist()#1 is feminine
        masc2=np.repeat(0, 26).tolist() #0 is masculine
        for i in masc2:
            test_classes.append(i) 
    elif testingset=='gender_stereotypes':
        test_word_list=['petite', 'cooking', 'graceful',  'housework', 'soft', 'whisper', 'flirtatious', 'accepting', 'blonde', 'blond', 'doll', 'dolls','nurse',  'estrogen', 'lipstick','pregnant', 'nanny', 'pink', 
                 'sewing', 'modeling', 'dainty', 'gentle', 'children','pregnancy', 'nurturing', 'depressed', 'nice', 'emotional','depression', 'home', 'kitchen', 'quiet', 'submissive',
                   'soldier', 'army', 'drafted', 'military',   'beard', 'mustache', 'genius', 'engineering', 'math', 
                  'brilliant', 'strong', 'strength',  'politician', 'programmer','doctor', 'sexual', 'aggressive', 
                    'testosterone', 'tall', 'competitive', 'big', 'powerful', 'mean', 'sports', 'fighting', 'confident', 'rough', 'loud', 'worldly',
                   'experienced', 'insensitive', 'ambitious', 'dominant']
        test_classes=np.repeat(1, 33 ).tolist() #1 is feminine
        masc2=np.repeat(0,33).tolist() #0 is masculine
        for i in masc2:
            test_classes.append(i) 
    else:
        print('choose a testing set: gender, moral, health, ses, or gender_stereotypes')
        
    test_words=[]
    test_word_list_checked=[]
    test_classes_checked=[] 
    for i in test_word_list:
        try:
            test_words.append(yourmodelhere[i])
            test_word_list_checked.append(i)
            test_classes_checked.append(test_classes[test_word_list.index(i)]) #new
        except KeyError:
            continue
            #print(str(i) +  ' was not in this Word2Vec models vocab, and has been removed as a test word')
            #index_missing= test_word_list.index(i)
            #del(test_classes[index_missing])
            #test_words.append(yourmodelhere[test_word_replacement])
            #test_word_list_checked.append(test_word_replacement)
            #get index of word, and remove this from classes, and do not append to list of vectors and word-list

    test_words= np.asarray(test_words)
    print('\033[1m'+ "Number of test words in model vocabulary, out of 60: " + '\033[0m' + str(len(test_words)))
    return(test_word_list_checked, test_words, test_classes_checked)
    
    
#a function to do a projection with a new word list, using a dimension learned on a training set of words
def do_projections(subspace, new_word_list, yourmodelhere):
    pos_word_list, neg_word_list, pos_words, neg_words, train_classes_pos, train_classes_neg = select_training_set(subspace, yourmodelhere) #could set this to your own training set, e.g., gender_2 or gender_3 to see how obesity results change with different training set
    test_word_list,test_words, test_classes = select_testing_set(subspace, yourmodelhere)
    directionVec= MEAN_get_directionVec_differences(pos_words, neg_words, yourmodelhere, train_index_list=None)
    
    #In a given model, we don't initially know the "direction" of a dimensions. In other words, we don't know whether a positive projection will be feminine or masculine. So, inductively find the correct labels 1/0. 
    #Consider this assumption that embedded in the code carefully when interpreting accuracy rates  
    pos_class=[] 
    for word in range(0, len(train_classes_pos)): #for word in this set of training words
        wordToProject=np.hstack(normalizeME(pos_words[word]))
        proj=project(wordToProject, directionVec)
        if proj > 0:
            pos_class.append(1) #evidence from this training word that the "positive words" are learned as positive
        elif proj < 0:
            pos_class.append(0) #evidence from this training word that the "positive words" are learned as negative
    
    #Now get predictions on the training set
    predictions_train=[]
    projections_train=[]
    combinedposnegtrain= np.concatenate([pos_words, neg_words], axis=0)
    train_classes=np.concatenate([train_classes_pos, train_classes_neg], axis=0) 
    if mode(pos_class)==1: #if the positive class is mostly 1, keep labels as is. Note that if the positive class is split exactly 50/50 this won't work, but also then accuracy is equal to flip of coin.    
        for word in range(0, len(combinedposnegtrain)):
            wordToProject=np.hstack(normalizeME(combinedposnegtrain[word]))
            proj=project(wordToProject, directionVec)
            projections_train.append(proj)
            if proj > 0:
                predictions_train.append(1)
            elif proj<0:
                predictions_train.append(0)
    elif mode(pos_class)==0: #if the positive class is mostly -1, reverse labels
        for word in range(0, len(combinedposnegtrain)):
            wordToProject=np.hstack(normalizeME(combinedposnegtrain[word]))
            proj=project(wordToProject, directionVec)
            projections_train.append(-(proj))
            if proj > 0:
                predictions_train.append(0)
            elif proj<0:
                predictions_train.append(1)
    else:
        print("No clear label")

    
    #Now get predictions on the testing set
    predictions_test=[]
    projections_test=[]
    if mode(pos_class)==1:
        for word in range(0, len(test_words)):
            wordToProject=np.hstack(normalizeME(test_words[word]))
            proj=project(wordToProject, directionVec)
            projections_test.append(proj)
            if proj > 0:
                predictions_test.append(1)
            elif proj< 0:
                predictions_test.append(0)
    elif mode(pos_class)==0:
        for word in range(0, len(test_words)):
            wordToProject=np.hstack(normalizeME(test_words[word]))
            proj=project(wordToProject, directionVec)
            projections_test.append(-(proj))
            if proj > 0:
                predictions_test.append(0)
            elif proj< 0:
                predictions_test.append(1)
    else:
        print("No clear label")

    trainacc=accuracy_score(train_classes, predictions_train)
    testacc=accuracy_score(test_classes, predictions_test)
    trainacc_N=accuracy_score(train_classes, predictions_train, normalize=False)
    testacc_N=accuracy_score(test_classes, predictions_test, normalize=False)

    print('\033[1m' + 'Percent- Training accuracy: '+ '\033[0m'+  str(trainacc) + '\033[1m' + " Testing accuracy: " + '\033[0m'+  str(testacc) )
    print('\033[1m' + 'Number- Training accuracy: ' + '\033[0m'+ str(trainacc_N) + '\033[1m' + " Testing accuracy: " + '\033[0m' + str(testacc_N))
    
    predictions_new_word_list=[]
    projections_new_word_list=[]
    if new_word_list is not None:
        if mode(pos_class)==1:
            for word in range(0, len(new_word_list)):
                wordToProject=np.hstack(normalizeME(yourmodelhere[new_word_list[word]]))
                proj=project(wordToProject, directionVec)
                projections_new_word_list.append(proj)
                if proj > 0:
                    predictions_new_word_list.append(1)
                elif proj< 0:
                    predictions_new_word_list.append(0)
        elif mode(pos_class)==0:
            for word in range(0, len(new_word_list)):
                wordToProject=np.hstack(normalizeME(yourmodelhere[new_word_list[word]]))
                proj=project(wordToProject, directionVec)
                projections_new_word_list.append(-(proj))
                if proj > 0:
                    predictions_new_word_list.append(0)
                elif proj< 0:
                    predictions_new_word_list.append(1)
        else:
            print("No clear label")
    
    return(np.concatenate([pos_word_list, neg_word_list], axis=0), predictions_train, projections_train, train_classes, test_word_list, predictions_test, projections_test, test_classes, predictions_new_word_list,  projections_new_word_list)
    
#a function run cross-validation on the training words 
def do_kfold(yoursubspace, yourmodelhere):

    pos_word_list, neg_word_list, pos_words, neg_words, train_classes_pos, train_classes_neg = select_training_set(yoursubspace, yourmodelhere)
    #cross validation
    kf= KFold(n_splits=len(pos_words), shuffle=True) 
    #n_splits written here is leave-one-wordpair-out cross validation, which is the maximum for n_splits. Try various quantities of n_splits. 
    #Note that this cross-validaton is by word-pair, so if we leave out one word-pair from the direction extraction process, 
    #we leave out two words on which to "test," giving two test predictions.
    #NOTE that we are only using our training words here, but dividing it into a "sub" training set, and an unseen set of the trainign words, we call in this code block the "test" set. But we do not use our unseen true "test" words until later on. 

    testacc=[]
    trainacc=[]

    for train_index, test_index in kf.split(pos_words): #only need the indices on pos words or neg words, then will be the same indices to use for both    
        directionVec= MEAN_get_directionVec_differences(pos_words, neg_words, yourmodelhere, train_index_list=train_index)
        #In a given model, we don't initially know the "direction" of a dimensions. In other words, we don't know whether a positive projection will be feminine or masculine. So, inductively find the correct labels 1/0. 
        #Consider this assumption that embedded in the code carefully when interpreting accuracy rates  
        #We learn this rule on the training model and then apply it on the fresh unseen testing model to make sure we're really detecting something.   
        pos_class=[] 
        for word in range(0, len(train_classes_pos[train_index])): #for word in this set of training words
            wordToProject=np.hstack(normalizeME(pos_words[train_index][word]))
            proj=project(wordToProject, directionVec)
            if proj > 0:
                pos_class.append(1) #evidence from this training word that the "positive words" are learned as positive
            elif proj < 0:
                pos_class.append(0) #evidence from this training word that the "positive words" are learned as negative
    
        #Now get predictions on the "training" set of the fold
        predictions_train=[]
        combinedposnegtrain= np.concatenate([pos_words[train_index] ,neg_words[train_index]], axis=0)
        if mode(pos_class)==1: #if the positive class is mostly 1, keep labels as is. Note that if the positive class is split exactly 50/50 this won't work, but also then accuracy is equal to flip of coin.    
            for word in range(0, len(combinedposnegtrain)):
                wordToProject=np.hstack(normalizeME(combinedposnegtrain[word]))
                proj=project(wordToProject, directionVec)
                if proj > 0:
                    predictions_train.append(1)
                elif proj<0:
                    predictions_train.append(0)
            classes_train= np.concatenate([train_classes_pos[train_index] , train_classes_neg[train_index]], axis=0)
        elif mode(pos_class)==0: #if the positive class is mostly -1, reverse labels
            for word in range(0, len(combinedposnegtrain)):
                wordToProject=np.hstack(normalizeME(combinedposnegtrain[word]))
                proj=project(wordToProject, directionVec)
                if proj > 0:
                    predictions_train.append(0)
                elif proj<0:
                    predictions_train.append(1)
            classes_train= np.concatenate([train_classes_pos[train_index] , train_classes_neg[train_index]], axis=0)
        else:
            print("No clear label")
    
        #Now get predictions on the "testing" set of the fold
        predictions_test=[]
        combinedposnegtest= np.concatenate([pos_words[test_index] ,neg_words[test_index]], axis=0)
        if mode(pos_class)==1:
            for word in range(0, len(combinedposnegtest)):
                wordToProject=np.hstack(normalizeME(combinedposnegtest[word]))
                proj=project(wordToProject, directionVec)
                if proj > 0:
                    predictions_test.append(1)
                elif proj< 0:
                    predictions_test.append(0)
            classes_test= np.concatenate([train_classes_pos[test_index], train_classes_neg[test_index]], axis=0)
        elif mode(pos_class)==0:
            for word in range(0, len(combinedposnegtest)):
                wordToProject=np.hstack(normalizeME(combinedposnegtest[word]))
                proj=project(wordToProject, directionVec)
                if proj > 0:
                    predictions_test.append(0)
                elif proj< 0:
                    predictions_test.append(1)
            classes_test= np.concatenate([train_classes_pos[test_index], train_classes_neg[test_index]], axis=0)
        else:
            print("No clear label")
        trainacc.append(accuracy_score(classes_train, predictions_train))
        testacc.append(accuracy_score(classes_test, predictions_test))
    
    print('\033[1m' +'Mean Accuracy across Training Subsets:'  + '\033[0m'+ str(mean(trainacc)))
    print('\033[1m' +'Standard Deviation of Accuracy across Training Subsets:'  + '\033[0m'+ str(stdev(trainacc)))
    print('\033[1m' +  'Mean Accuracy across Held-Out Subsets: ' + '\033[0m'+ str(mean(testacc)))
    print('\033[1m' +'Standard Deviation of Accuracy across Held-Out Subsets: ' + '\033[0m' + str(stdev(testacc)))