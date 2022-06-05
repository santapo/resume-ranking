import logging

def try_get_score(scoring_function, cv):
    try:
        return scoring_function(cv)
    except:
        logging.error("Error in scoring function: {}".format(scoring_function))
        return 0