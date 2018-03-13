
import re
class attack_level():
    def __init__(self,filename):
        self.file = filename
        self.tweet_list = self.get_file_data()
        self.injured_word = ('seriously','wounded','injured','critical','hurt','injuring','casualties',
                             'moderately','wounding','wounds','civilians','wound','missing')
        self.death_word = ('kills','killing','kill','dead','killed','martyrs')
        self.injured, self.death = self.get()
        self.level = self.get_level()

    def get_single_tweet_injured_death(self,tweet):
        word_list = tweet.split()
        injured_number = 0
        death_number = 0
        for i in range(1,len(word_list)-1):
            if word_list[i] in self.injured_word:
                try:
                    injured_number = self.spoken_word_to_number(word_list[i-1])
                except AssertionError:
                    try:
                        injured_number = self.spoken_word_to_number(word_list[i+1])
                    except AssertionError:
                        injured_number = 0
            if word_list[i] in self.death_word:
                try:
                    death_number = self.spoken_word_to_number(word_list[i-1])
                except AssertionError:
                    try:
                        death_number = self.spoken_word_to_number(word_list[i+1])
                    except AssertionError:
                        death_number = 0
            death_number_match = re.match('\d*\d$',str(death_number))
            death_number = death_number_match.group() if death_number_match else 0
            injured_number_match = re.match('\d*\d$',str(injured_number))
            injured_number = injured_number_match.group() if death_number_match else 0
            if injured_number == death_number:
                death_number = 0
        return {'death':death_number,'injured':injured_number}

    def get_all(self):
        injured_list = list()
        death_list = list()
        all_list = list()
        for tweet in self.tweet_list:
            single_tweet_injured_death =  self.get_single_tweet_injured_death(tweet)
            injured = single_tweet_injured_death['injured']
            death = single_tweet_injured_death['death']
            injured_list.append(injured)
            death_list.append(death)
        injured_dict = dict()
        death_dict = dict()
        for i in injured_list:
            injured_dict.update({i:0})
        for i in injured_list:
            injured_dict[i] +=1
        for i in death_list:
            death_dict.update({i: 0})
        for i in death_list:
            death_dict[i] += 1
        death_number = 0
        injured_number = 0
        max_freq = 0
        for death,freq in death_dict.items():
            if (death != 0 and death != '0') and freq > max_freq:
                death_number = int(death)
                max_freq = freq
        if max_freq == 1:
            death_number = 0
        max_freq = 0
        for injured,freq in injured_dict.items():
            if (injured != 0 and injured != '0') and freq > max_freq:
                injured_number = int(injured)
                max_freq = freq
        if max_freq == 1:
            injured_number = 0
        return {'injured':injured_number,'death':death_number}

    def get(self):
        all = self.get_all()
        injured = all['injured']
        death = all['death']
        return injured,death

    def get_level(self):
        all = 0
        level = ''
        if self.injured == self.death:
            all = self.death
        else:
            all = self.death + self.injured
        if all < 30:
            level = '一般事件'
        if all > 30 and all < 100:
            level = '较大事件'
        if all > 100 and all < 300:
            level = '重大事件'
        if all > 300:
            level = '特别重大事件'
        if level == '':
            level = 'Sorry, can not get level'
        return level

    def get_file_data(self):
        tweet_list = list()
        with open(self.file) as f:
            for line in f:
                line = line.lower()
                tweet_list.append(line)
        return tweet_list

    def spoken_word_to_number(self,n):
        number = re.match('\d*\d',n)
        if number:
            return number.group()
        _known = {
            'zero': 0,'one': 1,'two': 2,'three': 3,'four': 4,'five': 5,'six': 6,'seven': 7,'eight': 8,'nine': 9,
            'ten': 10,'eleven': 11,'twelve': 12,'thirteen': 13,'fourteen': 14,'fifteen': 15,'sixteen': 16,
            'seventeen': 17,'eighteen': 18,'nineteen': 19,'twenty': 20,'thirty': 30,'forty': 40,'fifty': 50,'sixty': 60,
            'seventy': 70,'eighty': 80,'ninety': 90
        }
        n = n.lower().strip()
        if n in _known:
            return _known[n]
        else:
            inputWordArr = re.split('[ -]', n)
        assert len(inputWordArr) > 1  # all single words are known
        # Check the pathological case where hundred is at the end or thousand is at end
        if inputWordArr[-1] == 'hundred':
            inputWordArr.append('zero')
            inputWordArr.append('zero')
        if inputWordArr[-1] == 'thousand':
            inputWordArr.append('zero')
            inputWordArr.append('zero')
            inputWordArr.append('zero')
        if inputWordArr[0] == 'hundred':
            inputWordArr.insert(0, 'one')
        if inputWordArr[0] == 'thousand':
            inputWordArr.insert(0, 'one')
        inputWordArr = [word for word in inputWordArr if word not in ['and', 'minus', 'negative']]
        currentPosition = 'unit'
        prevPosition = None
        output = 0
        for word in reversed(inputWordArr):
            if currentPosition == 'unit':
                number = _known[word]
                output += number
                if number > 9:
                    currentPosition = 'hundred'
                else:
                    currentPosition = 'ten'
            elif currentPosition == 'ten':
                if word != 'hundred':
                    number = _known[word]
                    if number < 10:
                        output += number * 10
                    else:
                        output += number
                # else: nothing special
                currentPosition = 'hundred'
            elif currentPosition == 'hundred':
                if word not in ['hundred', 'thousand']:
                    number = _known[word]
                    output += number * 100
                    currentPosition = 'thousand'
                elif word == 'thousand':
                    currentPosition = 'thousand'
                else:
                    currentPosition = 'hundred'
            elif currentPosition == 'thousand':
                assert word != 'hundred'
                if word != 'thousand':
                    number = _known[word]
                    output += number * 1000
            else:
                assert "Can't be here" == None
        return (output)

class disasters_level(attack_level):
    def __init__(self,filename):
        attack_level.__init__(self,filename)

    '''自然灾害之地震定义的特征有：1.地震等级 2.伤亡人数'''
    def extract_magnitude(self):
        possible_level = dict()
        for tweet in self.tweet_list:
            pattern = "[0-9][.][0-9]"
            match = re.search(pattern,tweet)
            if match:
                extract_level = match.group()
                if extract_level not in possible_level:
                    possible_level.update({extract_level:1})
                else:
                    possible_level[extract_level] += 1
        if len(possible_level) != 0:
            max_frequent_level = max(possible_level.items(),key = lambda x : x[1])
        else:
            max_frequent_level = None
        return max_frequent_level[0]

    def get_level(self):
        all = 0
        level = ''
        magnitude = float(self.extract_magnitude())
        print (magnitude)
        if self.injured == self.death:
            all = self.death
        else:
            all = self.death + self.injured
        if magnitude != None:
            if magnitude > 8.0:
                level = '特别重大事件'
            elif magnitude > 7.0:
                level = '重大事件'
            elif magnitude > 5.5:
                if all >= 200:
                    level = '重大事件'
                else:
                    level = '较大事件'
            else:
                if all >= 100:
                    level = '较大事件'
                else:
                    level = '一般事件'
        else:
            if all < 30:
                level = '一般事件'
            if all > 30 and all < 100:
                level = '较大事件'
            if all > 100 and all < 300:
                level = '重大事件'
            if all > 300:
                level = '特别重大事件'
        if level == '':
            level = 'Sorry, can not get level'
        return level

class accident_level(attack_level):
    def __init__(self,filename):
        attack_level.__init__(self,filename)

    def get_level(self):
       return self.get_level()

class military_level(attack_level):
    def __int__(self,filename):
        attack_level.__init__(self,filename)

    def get_level(self):
        return self.get_level()




if __name__ == '__main__':
    '''每一个类型事件定义了一个类，输入一个文件路径，调用.get_level()获得等级,.get()方法获得死亡和受伤人数'''
    dis = disasters_level(r'/home/nfs/yangl/event_detection/testdata/disasters/2018-3-taiwan-earthquake.txt')
    print (dis.get())
    print (dis.get_level())

