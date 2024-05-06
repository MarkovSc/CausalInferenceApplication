## NN Methods
import pandas, datetime, os, time,glob,argparse, numpy as np

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.losses import *
from tensorflow.keras.metrics import *


import keras
from keras import Input, layers, Model,regularizers
from keras.layers import Add, Embedding,Flatten,Multiply,Lambda,Concatenate,Dense,Subtract,Dropout,BatchNormalization
from keras.metrics import binary_accuracy
from tensorflow.keras.optimizers import SGD, Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau, TerminateOnNaN


from sklearn import preprocessing,linear_model
from sklearn.metrics import r2_score, mean_squared_error, mean_squared_log_error
from sklearn.model_selection import cross_val_score,KFold
from sklearn.preprocessing import LabelEncoder,FunctionTransformer,StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn import clone
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split


def check_inputs(Y, T, X):
    if(str(type(T)) == "<class 'pandas.core.frame.DataFrame'>"):
        T = T.to_numpy()
    if(str(type(Y)) == "<class 'pandas.core.frame.DataFrame'>"):
        Y = Y.to_numpy()
    return Y.flatten(), T.flatten(), X


class XLearnerDeep():
    def __init__(self, models=None,
                cate_models=None,
                propensity_model=None, dense_col=None, cat_col=None, scaler=None, is_regression=True, loss="mean_squared_error"):

        if(str(type(models)) == "<class 'function'>"):
            self.models = [models(dense_col,cat_col,is_regression=is_regression,is_relu_output=True, loss = loss), models(dense_col, cat_col,is_regression=is_regression,is_relu_output=True, loss = loss)]
        else:
            self.models = [clone(models, safe=False), clone(models, safe=False)]

        if(str(type(cate_models)) == "<class 'function'>"):
            self.cate_models = [cate_models(dense_col,cat_col,is_regression=True, loss = loss),cate_models(dense_col,cat_col,is_regression=True, loss = loss)]
        else:
            self.cate_models = [clone(cate_models, safe=False), clone(cate_models, safe=False)]

        self.propensity_model = LogisticRegression()
        self.scaler = scaler if(scaler is not None) else preprocessing.FunctionTransformer(func=None, inverse_func= None, validate=True)
        self.log_scaler = FunctionTransformer(func=np.log1p, inverse_func= np.expm1, validate=True)
        self.dense_col = dense_col
        self.cat_col = cat_col
        self.trainBatchSize = 256
        self.predictBatchSize = 4000

    def fit(self, Y, T, X, evaluate_inner = True,firstStageDebug=False):
        Y, T, X = check_inputs(Y, T, X)
        Y_norm = self.scaler.transform(Y.reshape(len(Y), -1)).flatten()
        ## 这里有个大内存操作
        if(str(type(self.models[0])) == "<class 'keras.engine.functional.Functional'>"):
            for ind in range(2):
                self.models[ind].fit([X[T == ind][self.dense_col], X[T == ind][self.cat_col]], Y_norm[T == ind], epochs=10, batch_size=self.trainBatchSize, verbose=1, validation_split =0.3, workers=40,  use_multiprocessing=True)
            
            mu_t = self.models[1].predict([X[T == 0][self.dense_col],X[T == 0][self.cat_col]], verbose=1,workers = 40, batch_size = self.predictBatchSize, use_multiprocessing=True)
            mu_c = self.models[0].predict([X[T == 1][self.dense_col],X[T == 1][self.cat_col]], verbose=1,workers = 40, batch_size = self.predictBatchSize, use_multiprocessing=True)
            mu_t = self.scaler.inverse_transform(mu_t.reshape(len(mu_t), -1)).flatten()
            mu_c = self.scaler.inverse_transform(mu_c.reshape(len(mu_c), -1)).flatten()

            mu_inner_t = self.models[1].predict([X[self.dense_col],X[self.cat_col]], verbose=1,workers = 40, batch_size = self.predictBatchSize, use_multiprocessing=True)
            mu_inner_c = self.models[0].predict([X[self.dense_col],X[self.cat_col]], verbose=1,workers = 40, batch_size = self.predictBatchSize, use_multiprocessing=True)
            mu_inner_t = self.scaler.inverse_transform(mu_inner_t.reshape(len(mu_inner_t), -1)).flatten()
            mu_inner_c = self.scaler.inverse_transform(mu_inner_c.reshape(len(mu_inner_c), -1)).flatten()


        else:
            for ind in range(2):
                self.models[ind].fit(X[T == ind], Y_norm[T == ind])
            mu_t = self.models[1].predict(X[T == 0])
            mu_c = self.models[0].predict(X[T == 1])
            mu_t = self.scaler.inverse_transform(mu_t.reshape(len(mu_t), -1)).flatten()
            mu_c = self.scaler.inverse_transform(mu_c.reshape(len(mu_c), -1)).flatten()

            mu_inner_t = self.models[1].predict(X)
            mu_inner_c = self.models[0].predict(X)
            mu_inner_t = self.scaler.inverse_transform(mu_inner_t.reshape(len(mu_inner_t), -1)).flatten()
            mu_inner_c = self.scaler.inverse_transform(mu_inner_c.reshape(len(mu_inner_c), -1)).flatten()
          
        
        imputed_effect_on_controls = mu_t.flatten() - Y[T == 0]
        imputed_effect_on_treated = Y[T == 1] - mu_c.flatten()

        #evaluate the mu_c result
        if(evaluate_inner):
            print("r2_score", r2_score(Y[T == 0], mu_t), r2_score(Y[T == 1], mu_c))
            print("mse", mean_squared_error(Y[T == 0], mu_t), mean_squared_error(Y[T == 1], mu_c))
            print("ate", np.mean(Y[T == 1]) - np.mean(Y[T==0]), np.mean(mu_t) - np.mean(mu_c))
            print("nusiance", np.mean(imputed_effect_on_controls), np.mean(imputed_effect_on_treated))

            uplift = mu_inner_t  - mu_inner_c
            eval_inner = evaluate_auuc_with_data(uplift, T, Y)
            print("Inner AUUC", np.mean(uplift), eval_inner[1:])

        if(firstStageDebug):
            return

        self.imputed_effect_on_controls= imputed_effect_on_controls
        self.imputed_effect_on_treated = imputed_effect_on_treated
        self.cate_mean_norm = (abs(np.mean(imputed_effect_on_controls)) +  abs(np.mean(imputed_effect_on_treated)))/2
        self.mean_scaler = FunctionTransformer(func = lambda x: x/self.cate_mean_norm,inverse_func=lambda x: x*self.cate_mean_norm, validate=True)
        self.mean_scaler= FunctionTransformer(func = None,inverse_func=None, validate=True)
        

        gc.collect()
        print("fit --- cate_model")
        if(str(type(self.cate_models[0])) == "<class 'keras.engine.functional.Functional'>"):
            imputed_effect_on_controls = self.mean_scaler.transform(imputed_effect_on_controls.reshape(len(imputed_effect_on_controls), -1)).flatten()
            imputed_effect_on_treated  = self.mean_scaler.transform(imputed_effect_on_treated.reshape(len(imputed_effect_on_treated), -1)).flatten()
       
            self.cate_models[0].fit([X[T == 0][self.dense_col],X[T == 0][self.cat_col]], imputed_effect_on_controls, epochs=10, batch_size=self.trainBatchSize, verbose=1, validation_split =0.3, workers=40,  use_multiprocessing=True)
            self.cate_models[1].fit([X[T == 1][self.dense_col],X[T == 1][self.cat_col]], imputed_effect_on_treated, epochs=10, batch_size=self.trainBatchSize, verbose=1, validation_split =0.3, workers=40,  use_multiprocessing=True)
        else:
            self.cate_models[0].fit(X[T == 0], imputed_effect_on_controls)
            self.cate_models[1].fit(X[T == 1], imputed_effect_on_treated)
        print("fit --- propensity_model")
        self.propensity_model.fit(X, T)
        gc.collect()

    def effect(self, X):
        m = X.shape[0]
        propensity_scores = self.propensity_model.predict_proba(X)[:, 1:]
        np.clip(propensity_scores, 0.3, 0.7, out=propensity_scores)

        if(str(type(self.cate_models[0])) == "<class 'keras.engine.functional.Functional'>"):
            tau_hat = propensity_scores * self.mean_scaler.inverse_transform(self.cate_models[0].predict([X[self.dense_col], X[self.cat_col]], verbose=1,workers = 10, batch_size = self.predictBatchSize, use_multiprocessing=True)).reshape(m, -1) \
                + (1 - propensity_scores) * self.mean_scaler.inverse_transform(self.cate_models[1].predict([X[self.dense_col], X[self.cat_col]], verbose=1,workers = 10, batch_size = self.predictBatchSize, use_multiprocessing=True)).reshape(m, -1)
            return tau_hat
        else:
            tau_hat = propensity_scores * self.cate_models[0].predict(X).reshape(m, -1) \
                + (1 - propensity_scores) * self.cate_models[1].predict(X).reshape(m, -1)
            return tau_hat


# define baseline model
def nn_model(dense_col, cat_col, is_regression=True, is_relu_output = False, loss="mean_squared_error"):
    inputs = Input(shape=(len(dense_col),))

    inputs_emb = Input(shape=(len(cat_col),))
    x_emb = Embedding(101, 1, input_length=len(cat_col))(inputs_emb)
    x_emb_flatten = Flatten()(x_emb)
    x_emb_flatten = Dense(1, activation=None)(x_emb_flatten)

    sum_kd_embed = Lambda(lambda x: K.sum(x, axis=1))(x_emb)
    square_sum_kd_embed = Multiply()([sum_kd_embed, sum_kd_embed])
    square_kd_embed = Multiply()([x_emb, x_emb]) 
    sum_square_kd_embed = Lambda(lambda x: K.sum(x, axis=1))(square_kd_embed)
    sub = Subtract()([square_sum_kd_embed, sum_square_kd_embed])
    sub = Lambda(lambda x: x*0.5)(sub)  
    second_order_sparse_layer = Lambda(lambda x: K.sum(x, axis=1, keepdims=True))(sub)
    second_order_sparse_layer = Dense(1, activation=None)(second_order_sparse_layer)
 
    x = inputs
    x = Concatenate(axis=1)([x,x_emb_flatten])
    x = Dense(256, activation='elu', kernel_regularizer=regularizers.l2(0.01))(x)
    
    for layer in [100, 100, 32]:
        x = Dense(layer, activation='elu', kernel_regularizer=regularizers.l2(0.01))(x)
    x = Concatenate(axis=1)([x,x_emb_flatten, second_order_sparse_layer])
    if(is_regression):
        if(is_relu_output):
            outputs = Dense(1, activation=tf.nn.relu)(x)
        else:
            outputs = Dense(1)(x)
        model = Model(inputs=[inputs,inputs_emb], outputs=outputs)
        model.compile(loss=loss, optimizer= tf.keras.optimizers.Adam(learning_rate=1e-3), metrics=['mean_absolute_error', 'mean_squared_error'])
        # model.compile(loss= tf.keras.losses.Huber(delta=1.0), optimizer= tf.keras.optimizers.Adam(learning_rate=1e-3), metrics=['mean_absolute_error', 'mean_squared_error'])
        
        return model
    else:
        outputs = Dense(1, activation=tf.nn.sigmoid)(x)
        model = Model(inputs=[inputs,inputs_emb], outputs=outputs)
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', 'binary_crossentropy'])
        return model


class CATE_Metrics(Callback):
    def __init__(self,YT,X, verbose=0):   
        super(CATE_Metrics, self).__init__()
        self.YT = YT
        self.X = X
        self.verbose=verbose
        YT_Value = YT.values
        self.ate_true = YT_Value[YT_Value[:,1]==1][:,0].mean() - YT_Value[YT_Value[:,1]==0][:,0].mean()

    def split_pred(self,concat_pred):
        #generic helper to make sure we dont make mistakes
        preds={}
        preds['y0_pred'] = concat_pred[:, 0]
        preds['y1_pred'] = concat_pred[:, 1]
        preds['t_pred'] = concat_pred[:, 2]
        preds['extra'] = concat_pred[:, 3:]
        return preds

    def ATE(self,concat_pred):
        p = self.split_pred(concat_pred)
        return p['y1_pred']-p['y0_pred']

    def on_epoch_end(self, epoch, logs={}):
        concat_pred=self.model.predict(self.X)
        #Calculate Empirical Metrics        
        ate_pred=tf.reduce_mean(self.ATE(concat_pred))
        tf.summary.scalar('ate', data=ate_pred, step=epoch)

        # auuc = tf.reduce_mean(evaluate_auuc_with_data(ate_pred, YT[:,1], YT[:,0]))
        # tf.summary.scalar('auuc', data=auuc, step=epoch)

        ate_err=tf.abs(self.ate_true-ate_pred); tf.summary.scalar('ate_err', data=ate_err, step=epoch)
        out_str=f' — ate_err: {ate_err:.4f}' 
        # out_str=f' — ate_err: {ate_err:.4f}  — auuc: {auuc:.4f}'   
        if self.verbose > 0: print(out_str)

class Base_Loss(Loss):
    #initialize instance attributes
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha
        self.name='standard_loss'

    def split_pred(self,concat_pred):
        #generic helper to make sure we dont make mistakes
        preds={}
        preds['y0_pred'] = concat_pred[:, 0]
        preds['y1_pred'] = concat_pred[:, 1]
        preds['t_pred'] = concat_pred[:, 2]
        preds['extra'] = concat_pred[:, 3:]
        return preds

    def treatment_bce(self, concat_true, concat_pred):
        t_true = concat_true[:, 1]
        t_pred = concat_pred[:, 2] 
        return tf.reduce_mean(K.binary_crossentropy(t_true, t_pred))

    def regression_loss(self, concat_true,concat_pred):
        y_true = concat_true[:, 0]
        t_true = concat_true[:, 1]
        p = self.split_pred(concat_pred)
        loss0 = tf.reduce_sum((1. - t_true) * tf.square(y_true - p['y0_pred']))
        loss1 = tf.reduce_sum(t_true * tf.square(y_true - p['y1_pred']))
        return loss0+loss1

    def regression_loss_mae(self,concat_true,concat_pred):
        y_true = concat_true[:, 0]
        t_true = concat_true[:, 1]
        p = self.split_pred(concat_pred)
        loss0 = tf.reduce_sum((1. - t_true) * tf.abs(y_true - p['y0_pred']))
        loss1 = tf.reduce_sum(t_true * tf.abs(y_true - p['y1_pred']))
        return loss0+loss1

    def standard_loss(self,concat_true,concat_pred):
        lossR = self.regression_loss(concat_true,concat_pred)
        lossP = self.treatment_bce(concat_true,concat_pred)
        return lossR + self.alpha * lossP

    #compute loss
    def call(self, concat_true, concat_pred):        
        return self.standard_loss(concat_true,concat_pred)

    def uplift_loss(concat_true, concat_pred):
        y_true = concat_true[:, 0]
        t_true = concat_true[:, 1]
        y0_pred = concat_pred[:, 0]
        y1_pred = concat_pred[:, 1]
        uplift_true = tf.reduce_mean(t_true * y_true - (1. - t_true) * y_true)
        uplift_pred = tf.reduce_mean(y1_pred - y0_pred)
        uplift_loss = tf.square(uplift_pred - uplift_true)
        return uplift_loss

    def dragonnet_loss_binarycross(self, concat_true, concat_pred):
        return self.regression_loss(concat_true, concat_pred) + self.treatment_bce(concat_true, concat_pred)

    def track_epsilon(self, concat_true, concat_pred):
        epsilons = concat_pred[:, 3]
        return tf.abs(tf.reduce_mean(epsilons))
     
    def tarreg_loss(self, concat_true, concat_pred, ratio=1.):
        vanilla_loss = self.dragonnet_loss_binarycross(concat_true, concat_pred)

        y_true = concat_true[:, 0]
        t_true = concat_true[:, 1]
        y0_pred = concat_pred[:, 0]
        y1_pred = concat_pred[:, 1]
        t_pred = concat_pred[:, 2]
        epsilons = concat_pred[:, 3]
        t_pred = (t_pred + 0.01) / 1.02
        t_pred = tf.clip_by_value(t_pred,0.3, 0.7,name='t_pred')
        y_pred = t_true * y1_pred + (1 - t_true) * y0_pred
        h = t_true / t_pred - (1 - t_true) / (1 - t_pred)
        y_pert = y_pred + epsilons * h
        targeted_regularization = tf.reduce_mean(tf.square(y_true - y_pert))
        # final
        loss = vanilla_loss + ratio * targeted_regularization
        return loss

class EpsilonLayer(layers.Layer):
    def __init__(self):
        super(EpsilonLayer, self).__init__()

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.epsilon = self.add_weight(name='epsilon',
                                       shape=[1, 1],
                                       initializer='RandomNormal',
                                       #  initializer='ones',
                                       trainable=True)
        super(EpsilonLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, inputs, **kwargs):
        # import ipdb; ipdb.set_trace()
        return self.epsilon * tf.ones_like(inputs)[:, 0:1]


def make_common_ty_component(inputs, input_cat,reg_l2=0.02):
    x_emb = Embedding(101, 1, input_length=inputs.shape[1])(input_cat)
    x_emb_flatten = Flatten()(x_emb)
    x_emb_flatten = Dense(1, activation=None)(x_emb_flatten)

    sum_kd_embed = Lambda(lambda x: K.sum(x, axis=1))(x_emb)
    square_sum_kd_embed = Multiply()([sum_kd_embed, sum_kd_embed])
    square_kd_embed = Multiply()([x_emb, x_emb]) 
    sum_square_kd_embed = Lambda(lambda x: K.sum(x, axis=1))(square_kd_embed)
    sub = Subtract()([square_sum_kd_embed, sum_square_kd_embed])
    sub = Lambda(lambda x: x*0.5)(sub)  
    second_order_sparse_layer = Lambda(lambda x: K.sum(x, axis=1, keepdims=True))(sub)
    second_order_sparse_layer = Dense(1, activation=None)(second_order_sparse_layer)
 
    # HYPOTHESIS
    phi = inputs
    for idx, dense_num in enumerate([200,200,200]):
        phi = Dense(units=dense_num, activation='elu', kernel_initializer='RandomNormal',name='phi_{0}'.format(idx))(phi)

    t_predictions = Dense(units=1,activation='sigmoid',name='t_prediction')(phi)
  
    y1_hidden, y0_hidden = phi, phi
    for idx, dense_num in enumerate([100,100,32]):
        y0_hidden = Dense(units=dense_num, activation='elu', kernel_regularizer=regularizers.l2(reg_l2),name='y0_hidden_{0}'.format(idx))(y0_hidden)
        y1_hidden = Dense(units=dense_num, activation='elu', kernel_regularizer=regularizers.l2(reg_l2),name='y1_hidden_{0}'.format(idx))(y1_hidden)
    
    y0_hidden = Concatenate(1)([y0_hidden, x_emb_flatten,second_order_sparse_layer])
    y1_hidden = Concatenate(1)([y1_hidden, x_emb_flatten,second_order_sparse_layer])

    y0_predictions = Dense(units=1, activation="relu", kernel_regularizer=regularizers.l2(reg_l2), name='y0_predictions')(y0_hidden)
    y1_predictions = Dense(units=1, activation="relu", kernel_regularizer=regularizers.l2(reg_l2), name='y1_predictions')(y1_hidden)

    return t_predictions, y0_predictions, y1_predictions


def make_dragonnet(dense_dim, cat_dim, reg_l2):
    inputs = Input(shape=(dense_dim,), name='input')
    inputs_cat = Input(shape=(cat_dim,), name='input_cat')

    # representation
    t_predictions, y0_predictions, y1_predictions = make_common_ty_component(inputs,inputs_cat,reg_l2)

    dl = EpsilonLayer()
    epsilons = dl(t_predictions, name='epsilon')
    
    CATE_loss=Base_Loss(alpha=10.0)
    concat_pred = Concatenate(1)([y0_predictions, y1_predictions, t_predictions, epsilons])
    model = Model(inputs=[inputs,inputs_cat], outputs=concat_pred)
    metrics = [CATE_loss,CATE_loss.regression_loss,CATE_loss.treatment_bce, CATE_loss.track_epsilon]
    model.compile(optimizer=Adam(lr=1e-3),loss=CATE_loss, metrics=metrics) #.tarreg_loss
    return model


def make_robust_effect(T,Y,Pred):
    mu0,mu1,ps = Pred[:,0],Pred[:,1],Pred[:,2]
    ps = np.clip(ps, 0.3, 0.7, out = ps)
    return T*(Y - mu1)/ps + mu1 - (1-T)*(Y - mu0)/(1-ps) + mu0