# lifespan init: ChatHandler.init called on app startup.
# get_user_input happy path: returns 200 with session_id, saves train (and optional test) using suffix from filenames, validates inputs, and calls store_session_in_db with correct args.
# get_user_input validation error: when validate_tabular_inputs returns error, respond 400 with error message.
# get_user_input server error: when store_session_in_db raises, respond 500 with error message.
# get_user_input no test file: handles test_file=None and stores test_path=None.
# get_user_input file naming: uses train{suffix}/test{suffix} defaults to .csv if missing.
# find_best_model no session: when get_session returns None, respond 404.
# find_best_model happy path: creates automl_data_path folder, calls AutoMLTrainer.train with DataFrames and correct target_column/time_limit, returns markdown string or raw object based on return type.
# find_best_model load_table calls: load_table invoked with correct paths for train and optional test.
# find_best_model response formatting: when leaderboard is DataFrame, to_markdown() is used; otherwise returned as-is.