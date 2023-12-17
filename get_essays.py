import pandas as pd


def getEssays(df):
    textInputDf = df[['id', 'activity', 'cursor_position', 'text_change']]
    textInputDf = textInputDf[textInputDf.activity != 'Nonproduction']
    valCountsArr = textInputDf['id'].value_counts(sort=False).values
    print(valCountsArr)
    lastIndex = 0
    essaySeries = pd.Series()
    for idx, valCount in enumerate(valCountsArr):
        currTextInput = textInputDf[[
            'activity', 'cursor_position', 'text_change']].iloc[lastIndex: lastIndex + valCount]
        lastIndex += valCount
        essayText = ""
        for Input in currTextInput.values:
            if Input[0] == 'Replace':
                replaceTxt = Input[2].split(' => ')
                essayText = essayText[:Input[1] - len(replaceTxt[1])] + replaceTxt[1] +\
                    essayText[Input[1] -
                              len(replaceTxt[1]) + len(replaceTxt[0]):]
                continue
            if Input[0] == 'Paste':
                essayText = essayText[:Input[1] - len(Input[2])] + \
                    Input[2] + essayText[Input[1] - len(Input[2]):]
                continue
            if Input[0] == 'Remove/Cut':
                essayText = essayText[:Input[1]] + \
                    essayText[Input[1] + len(Input[2]):]
                continue
            if "M" in Input[0]:
                croppedTxt = Input[0][10:]
                splitTxt = croppedTxt.split(' To ')
                valueArr = [item.split(', ') for item in splitTxt]
                moveData = (int(valueArr[0][0][1:]),
                            int(valueArr[0][1][:-1]),
                            int(valueArr[1][0][1:]),
                            int(valueArr[1][1][:-1]))
                if moveData[0] != moveData[2]:
                    if moveData[0] < moveData[2]:
                        essayText = essayText[:moveData[0]] + essayText[moveData[1]:moveData[3]] +\
                            essayText[moveData[0]:moveData[1]] + \
                            essayText[moveData[3]:]
                    else:
                        essayText = essayText[:moveData[2]] + essayText[moveData[0]:moveData[1]] +\
                            essayText[moveData[2]:moveData[0]] + \
                            essayText[moveData[1]:]
                continue
            essayText = essayText[:Input[1] - len(Input[2])] + \
                Input[2] + essayText[Input[1] - len(Input[2]):]
        print(essayText)
        essaySeries[idx] = essayText
    essaySeries.index = textInputDf['id'].unique()
    return pd.DataFrame(essaySeries, columns=['essay'])
