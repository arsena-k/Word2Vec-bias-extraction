# -*- coding: utf-8 -*-
"""
Created on Sun Jun 24 21:32:17 2018

@author: Alina Arseniev & Jacob Foster

Helper functions for Part D
"""

import numpy as np
from sklearn.metrics import accuracy_score
from gensim.models import Word2Vec,KeyedVectors
from sklearn.model_selection import KFold 
from statistics import mean, stdev
from sklearn import svm, datasets, decomposition, preprocessing



#a function to select the training words to find a dimensions
def select_training_set(trainingset, yourmodelhere): #options are: gender, moral, health, ses
    if trainingset=='gender':
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
        pos_word_replacement='woman' #here's the generic replacement for feminine words
        neg_word_replacement='man' #here's the generic replacement for masculine words
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
        pos_word_replacement='moral' #here's the generic replacement for moral words
        neg_word_replacement='immoral' #here's the generic replacement for immoral words
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
        pos_word_replacement='healthy' #here's the generic replacement for healthy words
        neg_word_replacement='ill' #here's the generic replacement for unhealthy words
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
                'blue_collar', 'low_income', 'neediest', 'less_fortunate', 'unsuccessful', 'economic_crisis', 'low_wage', 
                        'homeless']
        pos_word_replacement='wealthy' #here's the generic replacement for rich words
        neg_word_replacement='poor' #here's the generic replacement for poor words
    #gender is the training set of words used in corresponding paper to extract a gender dimension
    #gender_2 has fewer precise gender words like "he" vs "she," and some more noise since it also includes words that are gendered but less clearcut than the set "gender" above. We use gender_2 and gender_3 sets for robustness checks as described in our appendix.
    #gender_3 has even fewer precise gender words like "he" vs "she" than the set "gender_2" above,  and same added noise as "gender_2"
    elif trainingset=='gender_2':
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
    pos_words=[]
    neg_words=[]
    pos_word_list_checked=[] #we'll check that these training words are actually in the Word2Vec vocabulary. If a word isn't in the vocabulary, it gets replaced with a generic corresponding to the same concept (see pos_word_replacement and neg_word_replacement, above)
    neg_word_list_checked=[] #we'll check that these training words are actually in the Word2Vec vocabulary. If a word isn't in the vocabulary, it gets replaced with a generic corresponding to the same concept (see pos_word_replacement and neg_word_replacement, above)
    for i in pos_word_list:
        try:
            pos_words.append(yourmodelhere[i])
            pos_word_list_checked.append(i)
        except KeyError:
            #print(str(i) +  ' was not in this Word2Vec models vocab, and has been replaced with: ' + str(pos_word_replacement) ) #uncomment this to be alerted each time a pos training word-vector is replaced
            pos_words.append(yourmodelhere[pos_word_replacement])
            pos_word_list_checked.append(pos_word_replacement)
    for i in neg_word_list:
        try:
            neg_words.append(yourmodelhere[i])
            neg_word_list_checked.append(i)
        except KeyError:
            #print(str(i) +  ' was not in this Word2Vec models vocab, and has been replaced with: ' + str(neg_word_replacement) ) #uncomment this to be alerted each time a neg training word-vector is replaced
            neg_words.append(yourmodelhere[neg_word_replacement])
            neg_word_list_checked.append(neg_word_replacement)

    print('\033[1m' + "Number of pos train words: "+ '\033[0m' + str(len(pos_words)) + '\033[1m' + " Number of neg train words: " + '\033[0m' + str(len(neg_words)) )
    train_classes= np.concatenate((np.array(np.repeat(1, len(pos_words))), np.array(np.repeat(0, len(neg_words))))) #1 is feminine/moral/healthy/rich by default 0 is masculine/immoral/unhealthy/poor by default    
    words= np.concatenate((np.asarray(pos_words), np.asarray(neg_words)))
    words= preprocessing.normalize(np.asarray(words), norm='l2')
    pos_word_list_checked.extend(neg_word_list_checked) #pos_word_list now includes neg words
    
    return(pos_word_list_checked, words, train_classes)
    
    
    
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
        train_word_list_checked, train_words, train_classes= select_training_set('gender_3', yourmodelhere)  #gender testing set overfits since the words are too easy, so we use gender_3. See our paper appendix for more detail. 
    elif testingset=='moral':
        test_word_list= ['great', 'best', 'faith', 'chaste', 'wholesome', 'noble', 'honorable', 'immaculate', 'gracious', 
           'courteous', 'delightful', 'earnest', 'amiable', 'admirable', 'disciplined', 'patience', 'integrity',
            'restraint', 'upstanding', 'diligent', 'dutiful', 'loving', 'righteous','respectable', 'praise', 'devout', 'forthright',
            'depraved', 'repulsive', 'repugnant', 'corruption', 'vicious', 'unlawful', 'outrage',  'shameless', 'perverted',
            'filthy', 'lewd', 'subversive', 'sinister', 'murderous', 'perverse', 
           'monstrous', 'homicidal', 'indignant', 'misdemeanor', 'degenerate', 'malevolent', 'illegal','terrorist','terrorism',  
             'cheated', 'vengeful', 'culpable','vile', 'hateful', 'abuse', 'abusive', 'criminal', 'deviant']
        test_classes=np.repeat(1, 27 ).tolist() #1 is moral
        masc2=np.repeat(0,33).tolist() #0 is immoral
        for i in masc2:
            test_classes.append(i)
        train_word_list_checked, train_words, train_classes= select_training_set('moral', yourmodelhere)

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
        test_classes=np.repeat(1, 27).tolist() #1 is healthy
        masc2=np.repeat(0, 33 ).tolist() #0 is unhealthy
        for i in masc2:
            test_classes.append(i) 
        train_word_list_checked, train_words, train_classes= select_training_set('health', yourmodelhere)

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
        test_classes=np.repeat(1, 34).tolist()#1 is high ses
        masc2=np.repeat(0, 26).tolist() #0 is low ses
        for i in masc2:
            test_classes.append(i) 
        train_word_list_checked, train_words, train_classes= select_training_set('ses', yourmodelhere) 
    elif testingset=='gender_stereotypes': #extra set of development words to explore how sensitive the gender training word choice is to overfitting
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
        train_word_list_checked, train_words, train_classes= select_training_set('gender_3', yourmodelhere)   #as described in our paper appendix, gender overfits when we use the same set of words as the Bolukbasi and Larsen methods, so we use the set of words gender_3 
    else:
        print('choose a testing set: gender, moral, health, or ses')
    test_words=[]
    test_word_list_checked=[]
    test_classes_checked=[] 
    for i in test_word_list:
        try:
            test_words.append(yourmodelhere[i])
            test_word_list_checked.append(i)
            test_classes_checked.append(test_classes[test_word_list.index(i)]) 
        except KeyError:
            continue
            #print(str(i) +  ' was not in this Word2Vec models vocab, and has been removed as a test word') #uncomment this to be alerted each time a test word is not included in your model's vocabulary
            #index_missing= test_word_list.index(i) #new
            #del(test_classes[index_missing]) 
            #test_words.append(yourmodelhere[test_word_replacement])
            #test_word_list_checked.append(test_word_replacement)
            #get index of word, and remove this from classes, and do not append to list of vectors and word-list

    test_words= preprocessing.normalize(np.asarray(test_words), norm='l2')

    test_classes_checked=np.asarray(test_classes_checked)
    print('\033[1m'+ "Number of test words in model vocabulary, out of 60: " + '\033[0m' + str(len(test_words)))
    return(test_word_list_checked, test_words, test_classes_checked, train_word_list_checked, train_words, train_classes)
    
class subspaceselector:
    def __init__(self, direction):
        self.subspaceselection = direction
        if direction=='gender':
            self.pos_coded='Feminine'
            self.neg_coded='Masculine'
        if direction=='moral':
            self.pos_coded='Moral'
            self.neg_coded='Immoral'
        if direction=='health':
            self.pos_coded='Healthy'
            self.neg_coded='Unhealthy'
        if direction=='ses':
            self.pos_coded='High SES'
            self.neg_coded='Low SES'

#a function to run cross-validation on the training words 
def do_kfold(yoursubspace, yourmodelhere):

    train_word_list_checked, train_words, train_classes= select_training_set(yoursubspace, yourmodelhere)
    #cross validation
    kf= KFold(n_splits=len(train_words), shuffle=True)  
    #n_splits written here is leave-one-word-out cross validation, which is the maximum for n_splits. 
    #NOTE that we are only using our training words here, but dividing it into a "sub" training set, and an unseen set of the trainign words, we call in this code block the "test" set. But we do not use our unseen true "test" words until later on. 

    trainacc=[] 
    testacc=[] 

    for train_index, test_index in kf.split(train_words): #only need the indices on pos words or neg words, then will be the same indices to use for both    
    
        clf=svm.SVC(kernel='linear', C=1) #Use linear kernel, since not much data. More complex kernels performed worse and very high SD on accuracy. #C is 1 by default and it’s a reasonable default choice. If you have a lot of noisy observations you should decrease it. It corresponds to regularize more the estimation.
        #clf = RandomForestClassifier(n_estimators=100, max_depth=3, max_features=None, random_state=234) #max_features=None means all features are tried, rather than a sample. So the only randomness is the data. Default is that a sample of sqrt(n_features) is tried out for each tree, but this doesn't perform as well on training data, and doesn't make sense theoretically since I expect there is a few specific features that carry most of gender information. I tried betwen max depth of 2-4; 3 seems best for gender but on smaller sets consider 2.
        #clf = MLPClassifier(hidden_layer_sizes=(5)) #to try neural network rather than SVM, but really not enough data here, its just a sample of how to change the ML classifier here. 
        clf= clf.fit(train_words[train_index], train_classes[train_index] )
    
        #Now get predictions on the "training" set of the fold
        predictions_training=clf.predict(train_words[train_index])
        trainacc.append(accuracy_score(train_classes[train_index], predictions_training)) #append accuracy from this specific training subset

        #Now get predictions on subset of unseen trainning words (i.e., validation set)
        predictions_testing=clf.predict(train_words[test_index])
        testacc.append(accuracy_score(train_classes[test_index], predictions_testing)) #append accuracy from this specific 'testing' subset

    
    print('\033[1m' +'Mean Accuracy across Training Subsets:'  + '\033[0m'+ str(mean(trainacc)))
    print('\033[1m' +'Standard Deviation of Accuracy across Training Subsets:'  + '\033[0m'+ str(stdev(trainacc)))
    print('\033[1m' +  'Mean Accuracy across Held-Out Subsets: ' + '\033[0m'+ str(mean(testacc)))
    print('\033[1m' +'Standard Deviation of Accuracy across Held-Out Subsets: ' + '\033[0m' + str(stdev(testacc)))