#%%
import os
import pickle
import pprint
import numpy as np
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
from multiprocessing import Pool


def neighbor_check(x, y, cars_pos):
    neighbor_array = np.zeros(25, dtype=int)
    new_x = (x//120) * 120 # makes car position is right at the top left corner of the grid
    new_y = (y//50) * 50 ### - 15
    for i, car in enumerate(cars_pos):
        neighbor = [(car[0], car[1]), (car[0],car[1]+30), (car[0]+60, car[1]), (car[0]+60, car[1]+30)]
        for pair in neighbor:
            c = -1 # represents the column where it stays
            n_x = pair[0]
            n_y = pair[1]
            if n_x < 0:
                continue
            if n_x < new_x:
                if (new_x - n_x)//120 == 0: # last column
                    c = 0
            else:
                if (n_x - new_x)//120 == 0: # same column
                    c = 1
                elif (n_x - new_x)//120 == 1: # next column
                    c = 2
                elif (n_x - new_x)//120 == 2: # next of next column
                    c = 3
                elif (n_x - new_x)//120 == 3:
                    c = 4

            if c == 0 or c == 1 or c == 2 or c == 3 or c == 4:
                if n_y < new_y:
                    if (new_y - n_y)//50 == 0: # last row
                        neighbor_array[c+5] = 1
                    elif (new_y-n_y)//50 == 1: # last of last row
                        neighbor_array[c] = 1
                else:
                    if (n_y - new_y)//50 == 0: # same row
                        neighbor_array[c+10] = 1
                    elif (n_y - new_y)//50 == 1: # next row
                        neighbor_array[c+15] = 1
                    elif (n_y - new_y)//50 == 2:
                        neighbor_array[c+20] = 1
            else:
                continue

    if(y<=150): #在第一車道及最後一車道避免撞牆
        for i in range(5):
            neighbor_array[i] = 1
    elif(y>=500):
        for i in range(20,25):
            neighbor_array[i] = 1

    return neighbor_array


# 將資料從pickle中取出
path = os.path.join(os.path.dirname(__file__),"..","log")
allFile = os.listdir(path)
pickleSet = []
for file in allFile:
    with open(os.path.join(path,file),"rb") as f:
        pickleSet.append(pickle.load(f)) 

neighbor_pickle = [] #以pickle為單位
status_pickle = []

for aPickle in pickleSet:
    counter = [] #記index
    for idx, command in enumerate(aPickle['1P']['command'][:-10]):
        if command == ['MOVE_LEFT'] and aPickle['1P']['command'][idx-1] != ['MOVE_LEFT']:
            counter.append(idx-5)
        elif command != ['MOVE_LEFT'] and aPickle['1P']['command'][idx-1] == ['MOVE_LEFT']:
            counter.append(idx-1)

    
    for i in range(len(counter)//2): #只看左之下 沒撞到的
        neighbor_list = []
        status_list = []
        for sceneInfo in aPickle['1P']['scene_info'][counter[2*i]:counter[2*i+1]]:
            if sceneInfo['status'] == 'GAME_ALIVE':
                neighbor_list.append(neighbor_check(sceneInfo['x'], sceneInfo['y'], sceneInfo['all_cars_pos']))
                status_list.append(0)
        if len(neighbor_list) != 0:
            neighbor_pickle.append(neighbor_list)
            status_pickle.append(status_list)

    neighbor_list_ = []
    status_list_ = []
    for idx, sceneInfo in enumerate(aPickle['1P']['scene_info'][:]): #看撞到的
        if sceneInfo['status'] == 'GAME_OVER':
            if sceneInfo['y'] -  aPickle['1P']['scene_info'][idx-2]['y'] < 0:#左邊撞到
                for sceneInfo in aPickle['1P']['scene_info'][idx-5:]:
                    neighbor_list_.append(neighbor_check(sceneInfo['x'], sceneInfo['y'], sceneInfo['all_cars_pos']))
                    status_list_.append(1) #左邊撞到
    if len(neighbor_list_) != 0:
        neighbor_pickle.append(neighbor_list_)
        status_pickle.append(status_list_)

#%% 將資料排成相應格式
frames = 5
featureCount = 25

# x = np.array([(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0)]) # 25個
# y = np.array([0])
# for apickle in neighbor_pickle:
#     for frame_data in apickle:
#         x = np.vstack([x, frame_data])
# for apickle in status_pickle:
#     for frame_data in apickle:
#         y = np.vstack([y, frame_data])
# x = x[1::]
# y = y[1::]

# new_x = []
# new_y = []
# for i in range(len(x)-frames+1):
#     new_x.append(x[i:i+frames])
#     new_y.append(y[i+frames-1])
# new_x = np.array(new_x)
# new_y = np.array(new_y)
# new_x = np.reshape(new_x,(len(new_x),len(new_x[0])*len(new_x[0][0])))


# #相同training data刪掉(以一組feature+label為單位)
# dataSet = np.hstack([new_x,new_y]) #合併feature及target, 以利判斷是否重複  
# print("original",dataSet.shape)
# dataSet = np.unique(dataSet, axis=0) #把dataSet的相同列刪掉 
# print("non-repetitive",dataSet.shape)


dataSet_pickle = np.empty(frames*featureCount+1)
dataSet = np.zeros(frames*featureCount+1)
for i in range(len(neighbor_pickle)):
    x = np.array([(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0)]) # 25個
    y = np.array([0])
    for j in range(len(neighbor_pickle[i])): #將每個pickle檔的neighbor先疊好
        x = np.vstack((x,[neighbor_pickle[i][j][0],neighbor_pickle[i][j][1],neighbor_pickle[i][j][2],neighbor_pickle[i][j][3],neighbor_pickle[i][j][4],neighbor_pickle[i][j][5],
                neighbor_pickle[i][j][6],neighbor_pickle[i][j][7],neighbor_pickle[i][j][8],neighbor_pickle[i][j][9],neighbor_pickle[i][j][10],neighbor_pickle[i][j][11],neighbor_pickle[i][j][12],neighbor_pickle[i][j][13],
                neighbor_pickle[i][j][14],neighbor_pickle[i][j][15],neighbor_pickle[i][j][16],neighbor_pickle[i][j][17],neighbor_pickle[i][j][18],
                neighbor_pickle[i][j][19],neighbor_pickle[i][j][20],neighbor_pickle[i][j][21],neighbor_pickle[i][j][22],neighbor_pickle[i][j][23],neighbor_pickle[i][j][24]])) #一定要有X, 否則疊不起來 dataset_group[i][0],dataset_group[i][1],
        y = np.vstack((y,[status_pickle[i][j]]))
    x = x[1::]
    y = y[1::]
    X = np.empty([len(x)-(frames-1)-1,frames*featureCount])
    Y = np.empty([len(x)-(frames-1)-1,1],dtype = int)
    for i in range(len(x)-(frames-1)-1): #整理成每多個frames為一列的格式
        for j in range(frames):
            for k in range(featureCount): 
                X[i, featureCount * j + k] = x[i + j, k]
    for i in range(len(y)-(frames-1)-1):
        Y[i,0] = y[i+frames]
    dataSet_pickle = np.hstack([X,Y]) #合併feature及target, 以利判斷是否重複
    dataSet = np.vstack((dataSet, dataSet_pickle)) #把大X,Y放到dataSet 
dataSet = dataSet[1::]
print("original",dataSet.shape)
dataSet = np.unique(dataSet, axis=0) #把dataSet的相同列刪掉 
print("non-repetitive",dataSet.shape)

#刪掉部分沒撞到的資料
dataSet_dead = dataSet[dataSet[:,-1]!=0]
dataSet_alive = dataSet[dataSet[:,-1]==0]
alive = dataSet_alive.shape[0] #沒撞到資料的筆數
alive = int(alive * 0.1) #留下?成
np.random.shuffle(dataSet_alive)
dataSet_alive_stay = dataSet_alive[0:alive,:]
dataSet = np.concatenate((dataSet_dead,dataSet_alive_stay),axis=0)

X, Y = np.hsplit(dataSet, [frames*featureCount])
Y = Y.ravel()
# print(Y)


# 比例
dead_l = 0
dead_r = 0
alive_ = 0

for status in Y:
    if status == 1:
        dead_l += 1
    else:
        alive_ += 1

total = (alive_ + dead_l)
print("alive=",alive_," dead_left=",dead_l)
print("alive=%.0f" % ((alive_/total)*100)," dead_l=%.0f" % ((dead_l/total)*100))

#%%
# training
""" KNN """
x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.2)
k_final = 0
Accuracy = 0
for k in range(0,30):
    k = k+1
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(x_train,y_train)
    y_predict = model.predict(x_test)
    print("k = ",k ,"Accuracy = %.2f" % accuracy_score(y_predict,y_test))
    if accuracy_score(y_predict,y_test) > Accuracy:
        Accuracy = accuracy_score(y_predict,y_test)
        k_final = k
        
model = KNeighborsClassifier(n_neighbors=k_final)
model.fit(x_train,y_train)
y_predict = model.predict(x_test)
accuracy = float('{:.3f}'.format(accuracy_score(y_predict,y_test))) #分對的比例
training_score = float('{:.3f}'.format(model.score(x_train,y_train))) #mean accuracy
testing_score = float('{:.3f}'.format(model.score(x_test,y_test)))
print("k = ",k_final ,"Accuracy = ",accuracy)
print("training data score = ",training_score)
print("testing data score = ",testing_score)



# """ Decision tree """
# # x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.2)
# # model = tree.DecisionTreeClassifier(criterion='entropy', max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features=None, random_state=None)
# # gs = GridSearchCV(model, param_grid = {'max_depth': range(1, 30),'min_samples_split': range(5, 65, 10)}, scoring=None)
# # gs.fit(x_train,y_train)
# # bestParams = gs.best_params_
# # predict = gs.predict(x_test)
# # training_score = float('{:.3f}'.format(gs.score(x_train,y_train)))
# # testing_score = float('{:.3f}'.format(gs.score(x_test,y_test)))

# # print("best params = ", bestParams)
# # print("training score = ", training_score)
# # print("testing score = ", testing_score)

# """ Ramdom Forest or SVM """



# save
path = os.path.dirname(__file__)
path = os.path.join(path,"save")
if not os.path.isdir(path):
    os.mkdir(path)
with open(os.path.join(os.path.dirname(__file__),'save','KNN_data={}_frames={}_acc={}_k={}_hitOrNot_L_25SquareRev.pickle'.format(len(dataSet),frames,testing_score, k_final)),'wb') as f:
    pickle.dump(model,f)
