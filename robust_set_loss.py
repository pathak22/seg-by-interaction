import numpy as np


def robust_set_loss(logProbsNet, gtMasks, step=0.1, iouTh=0.75, maxIter=25,
                    ind=None):
    '''
    logProbsNet: logprobs, i.e, the output of last layer of the network being
                 trained. Should be of the shape [bs,2,ht,wt].
    gtMasks: ground truth mask. Should be of shape [bs,1,ht,wt]. It contains
             either 0 (background) or 1 (foreground).
    step (scalar): granularity of optimization (the lower it is, the finer the
                   results will be).
    iouTh (scalar): lower bound constraint on IoU overlap constraint
    maxIter (scalar): limit on how hardly the constraint should be enforced.
                    Higher means more emphasis on satisfying the constraint.
    output newMasks: new ground truth mask to be trained against.
                    Shape: [bs,1,ht,wt]. 0 (background) or 1 (foreground).
    '''
    if type(step) is np.ndarray:  # tensorflow hacking
        step = step[0]
        iouTh = iouTh[0]
        maxIter = maxIter[0]

    cIn = np.zeros((gtMasks.shape[0],), dtype=np.float)
    cOut = np.zeros((gtMasks.shape[0],), dtype=np.float)
    logProbs = np.copy(logProbsNet)
    indexer = np.zeros(gtMasks.shape, dtype=bool)

    for i in range(maxIter):
        iou_orig = iou(np.argmax(logProbs, axis=-1), gtMasks)
        unconverged = np.logical_not(iou_orig > iouTh)
        if np.all(np.logical_not(unconverged)):
            break

        indexer *= False
        indexer[unconverged] = gtMasks[unconverged] > 0.5
        logProbs[..., 1][indexer] += step
        iou_upIn = iou(np.argmax(logProbs, axis=-1), gtMasks)

        logProbs[..., 1][indexer] -= step
        indexer *= False
        indexer[unconverged] = gtMasks[unconverged] < 0.5
        logProbs[..., 1][indexer] -= step
        iou_downOut = iou(np.argmax(logProbs, axis=-1), gtMasks)

        indexer *= False
        indexer[unconverged] = gtMasks[unconverged] > 0.5
        logProbs[..., 1][indexer] += step
        iou_upInDownOut = iou(np.argmax(logProbs, axis=-1), gtMasks)

        improvedIn = np.logical_and(iou_upIn > iou_orig, unconverged)
        cIn[improvedIn] += step
        indexer *= False
        indexer[improvedIn] = gtMasks[improvedIn] < 0.5
        logProbs[..., 1][indexer] += step

        improvedOut = np.logical_and(
            np.logical_not(improvedIn),
            np.logical_and(iou_downOut > iou_orig, unconverged))
        cOut[improvedOut] += step
        indexer *= False
        indexer[improvedOut] = gtMasks[improvedOut] < 0.5
        logProbs[..., 1][indexer] -= step

        improvedInOut = np.logical_and(
            np.logical_not(improvedIn + np.logical_and(
                iou_downOut > iou_orig, unconverged)),
            unconverged)
        cIn[improvedInOut] += step
        cOut[improvedInOut] += step

        if ind is not None and unconverged[ind]:
            print("Iter %02d: iouIn=%.2f iouOut=%.2f iouInOut=%.2f " % (i,
                    iou_upIn[ind], iou_downOut[ind], iou_upInDownOut[ind],) +
                "iouOrig=%.2f cIn=%.1f cOut=%.1f" % (iou_orig[ind], cIn[ind],
                    cOut[ind]))

    if ind is not None:
        print('iou at convergence=%.2f' % iou_orig[ind])
    newMasks = np.argmax(logProbs, axis=-1).astype(np.int32)
    return newMasks


def iou(mask1, mask2):
    axis = (1, 2) if len(mask1.shape) > 2 else None
    union = np.sum(np.logical_or(mask1, mask2), axis=axis).astype('float')
    if axis is None and union == 0:
        return 0.0
    elif axis is not None:
        union[union == 0] += 1e-12
    intersection = np.sum(
        np.logical_and(mask1, mask2), axis=axis).astype('float')
    return intersection/union
