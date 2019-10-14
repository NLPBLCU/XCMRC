import numpy as np
import tensorflow as tf
MIN_NUM = -1e9

class Evaluator(object):
    def __init__(self, config, model, word2id):
        self.config = config
        self.model = model
        self.word_dict = { word2id[w]:w for w in word2id.keys()}

    def get_evaluation_from_batches(self, sess, batches):
        res = {}
        #import pdb
        #pdb.set_trace()
        acc = 0
        bs = 0
        total_size = 0
        for batch_id, batch in enumerate(batches):
            total_size += len(batch['doc'])
            bs = len(batch['doc'])
            answer_mask = np.zeros_like(batch['doc']).astype(np.float32) + MIN_NUM
            labels = np.zeros_like(batch['doc']).astype(np.float32)
            batch_true = []
            for i, items in enumerate(batch['doc']):
                tmp = []
                for j, item in enumerate(items):
                    if item in batch['candidates'][i]:
                        answer_mask[i][j] = 0
                    if item == batch['candidates'][i][int(batch['y'][i][0])]:
                        labels[i, j] = 1
                        tmp.append(j)
                assert tmp!=[]
                batch_true.append(tmp)
            feed_dict = {}
            feed_dict[self.model.doc] = batch['doc']
            feed_dict[self.model.y] = labels
            feed_dict[self.model.doc_mask] = batch['doc_mask']
            feed_dict[self.model.answer_mask] = answer_mask
            feed_dict[self.model.que] = batch['que']
            feed_dict[self.model.que_mask] = batch['que_mask']
            feed_dict[self.model.is_train] = False
            m_pred = sess.run(self.model.prediction, feed_dict=feed_dict)
            pred = np.argmax(m_pred, 1)
            for i, id in enumerate(batch['ids']):
                can = []
                for c in batch['candidates'][i]:
                    can.append(self.word_dict[c])
                res[id] = {'pred': self.word_dict[batch['doc'][i][pred[i]]],
                           'true': can[int(batch['y'][i][0])], 'can': can}
                if pred[i] in batch_true[i]:
                    acc += 1
        acc = acc / total_size
        return {'acc': acc, 'res': res}

