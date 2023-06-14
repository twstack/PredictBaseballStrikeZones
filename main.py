import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from pybaseball import statcast_batter, playerid_lookup


def run_svm(player_name, start_date='2008-04-01', end_date='2017-07-15'):
    # Lookup for player id
    player_id = playerid_lookup(player_name.split(' ')[1], player_name.split(' ')[0])['key_mlbam'].values[0]

    # Fetch player data
    player_df = statcast_batter(start_date, end_date, player_id)

    print(player_df['player_name'][0], ":\n")

    fig, ax = plt.subplots()

    # 4. Change 'S' to 1 and 'B' to 0 in the type column
    player_df['type'] = player_df['type'].map({'S': 1, 'B': 0})

    # 7. Remove NaN values from the 'plate_x', 'plate_z', and 'type' columns
    player_df = player_df.dropna(subset=['plate_x', 'plate_z', 'type', 'strikes'])

    # 9. Split the data into training and validation sets
    training_set, validation_set = train_test_split(player_df, random_state=1)

    # 10. Create the SVM classifier
    classifier = SVC(kernel='rbf')

    # 11. Fit the SVM to the training data
    classifier.fit(training_set[['plate_x', 'plate_z', 'strikes']], training_set['type'])

    # 8. Scatter plot
    plt.scatter(x=player_df['plate_x'], y=player_df['plate_z'], c=player_df['type'], cmap=plt.cm.coolwarm, alpha=0.25)

    # 12. Draw the decision boundary (comment this line if you are using more than 2 features)
    # draw_boundary(ax, classifier)
    ax.set_ylim(-2, 6)
    ax.set_xlim(-3, 3)

    # 13. Score the SVM on the validation data
    print('Validation accuracy: ',
          classifier.score(validation_set[['plate_x', 'plate_z', 'strikes']], validation_set['type']))

    # 14. Overfit the data and score again
    classifier = SVC(kernel='rbf', gamma=100, C=100)
    classifier.fit(training_set[['plate_x', 'plate_z', 'strikes']], training_set['type'])
    print('Overfitted validation accuracy: ',
          classifier.score(validation_set[['plate_x', 'plate_z', 'strikes']], validation_set['type']))

    # 15. Loop through different values of gamma and C for accuracy improvement
    best_score = 0
    best_params = {'gamma': None, 'C': None}
    for gamma in range(1, 10):
        for C in range(1, 10):
            classifier = SVC(kernel='rbf', gamma=gamma, C=C)
            classifier.fit(training_set[['plate_x', 'plate_z', 'strikes']], training_set['type'])
            score = classifier.score(validation_set[['plate_x', 'plate_z', 'strikes']], validation_set['type'])
            if score > best_score:
                best_score = score
                best_params['gamma'] = gamma
                best_params['C'] = C
    print('Best validation accuracy: ', best_score)
    print('Best parameters: ', best_params, "\n")

    plt.show()


run_svm('Aaron Judge')
run_svm('Jose Altuve')
run_svm('David Ortiz')
