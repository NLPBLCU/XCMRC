import numpy as np
import tensorflow as tf
MIN_NUM = -1e9

class Evaluator(object):
    def __init__(self, config, model, word2id):
        self.config = config
        self.model = model
        self.word_dict = {word2id[w]:w for w in word2id.keys()}

    def get_evaluation_from_batches(self, sess, batches):
        y_true = []
        y_pred = []
        res = {}
        import pdb
        #pdb.set_trace()
        for j, batch in enumerate(batches):
            bs = len(batch['doc'])
            feed_dict =  {self.model.doc: batch['doc'],
                     self.model.doc_mask: batch['doc_mask'],

                     self.model.que: batch['que'],
                     self.model.que_mask: batch['que_mask'],

                     self.model.y: np.reshape(batch['y'], [len(batch['y'])]).astype('int32'),

                     self.model.c: batch['candidates'],
                     self.model.is_train: False}

            m_pred = sess.run(self.model.prediction, feed_dict=feed_dict)
            y_true += batch['y']
            pred = np.argmax(m_pred, 1)
            y_pred += list(pred)
            for i, sample_id in enumerate(batch['ids']):
                can = []
                for c in batch['candidates'][i]:
                    can.append(self.word_dict[c])
                res[sample_id] = {'pred': can[pred[i]], 'true': can[int(batch['y'][i][0])], 'can': can}
        acc = 0
        assert len(y_pred) == len(y_true)
        for i in range(len(y_true)):
            if int(y_true[i][0]) == int(y_pred[i]):
                acc += 1
        acc = acc / len(y_true)
        return {'acc': acc, 'res': res}

