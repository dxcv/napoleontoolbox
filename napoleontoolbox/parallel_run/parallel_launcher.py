#!/usr/bin/env python3
# coding: utf-8

import sqlite3
import pandas as pd

from multiprocessing import Pool

class ParalleLauncher():


    def __init__(self,  db_path, meta_normalize=[True], meta_advance_features=[False], meta_advance_signals=[False], meta_history=[False], meta_layers=[[2048,1024,512,256,128,64,32]], convolutions=[0], activations=['relu'], meta_n=[126], meta_seeds=[0], epochss=[1], ss=[21], meta_n_past_features=[21], utilities=['f_drawdown']):
        print('Epochs ' + str(epochss))
        print('Seed ' + str(meta_seeds))
        print('Training period size ' + str(meta_n))
        print('Neural net layout ' + str(meta_layers))
        print('Advanced ' + str(meta_normalize))
        print('Advanced ' + str(meta_advance_features))
        print('Advanced ' + str(meta_advance_signals))
        print('History ' + str(meta_history))
        print('Normalized ' + str(meta_normalize))
        print('Rebalancing ' + str(ss))
        print('Activation ' + str(activations))
        print('Convolution' + str(convolutions))
        print('Utility' + str(utilities))
        print('n past features ' + str(meta_n_past_features))
        print('Rebalancing ' + str(ss))

        self.args = []
        self.counter = 1
        for utility in utilities:
            for n_past_features in meta_n_past_features:
                for convolution in convolutions:
                    for activation in activations:
                        for s in ss:
                            for epochs in epochss:
                                for normalize in meta_normalize:
                                    for whole_history in meta_history:
                                        for advance_feature in meta_advance_features:
                                            for advance_signal in meta_advance_signals:
                                                for seed in meta_seeds:
                                                    for layers in meta_layers:
                                                        for n in meta_n:
                                                            self.args.append((self,seed, utility, layers, epochs,
                                                                         n_past_features, n, s, whole_history,
                                                                         advance_feature, advance_signal, normalize,
                                                                         activation, convolution))
                                                            # self.args.append((seed, utility, param, layers, epochs,
                                                            #              n_past_features, n, s, whole_history,
                                                            #              advance_feature, advance_signal, normalize,
                                                            #              activation, convolution))
                                                            self.counter = self.counter + 1

        self.db_path = db_path
        self.runs = []
        self.totalRow = 0
        self.instantiateTable()
        self.empty_runs_to_investigate = []

    def launchParallelPool(self, toRun, use_num_cpu):
        with Pool(processes=use_num_cpu) as pool:
            results = pool.starmap(toRun, self.args)
            print('parallel computation done')

    def launchSequential(self, toRun):
        for meArg in self.args:
            toRun(*meArg)

    def instantiateTable(self):
        sqliteConnection = None
        try:
            sqliteConnection = sqlite3.connect(self.db_path)
            sqlite_create_table_query = '''CREATE TABLE parallel_run (
                                        effective_date date PRIMARY KEY);'''
            cursor = sqliteConnection.cursor()
            print("Successfully Connected to SQLite")
            cursor.execute(sqlite_create_table_query)
            sqliteConnection.commit()
            print("SQLite table created")
            cursor.close()
        except sqlite3.Error as error:
            print("Error while creating a sqlite table", error)
        finally:
            if (sqliteConnection):
                sqliteConnection.close()
                print("sqlite connection is closed")

    def addRun(self,run):
        sqliteConnection = None
        try:
            sqliteConnection = sqlite3.connect(self.db_path)
            sqlite_create_table_query = 'alter table parallel_run add column ' + run + ' real'
            cursor = sqliteConnection.cursor()
            print("Successfully Connected to SQLite")
            cursor.execute(sqlite_create_table_query)
            sqliteConnection.commit()
            print("SQLite table created")
            cursor.close()
            self.runs.append(run)
        except sqlite3.Error as error:
            print("Error while creating a sqlite table", error)
        finally:
            if (sqliteConnection):
                sqliteConnection.close()
                print("sqlite connection is closed")  # adding the column

    def insertResults(self, run, values):
        sqliteConnection = None
        try:
            success = True
            sqliteConnection = sqlite3.connect(self.db_path)
            cursor = sqliteConnection.cursor()
            for i, v in values.iteritems():
                print("Connected to SQLite")
                sqlite_insert_query = """INSERT INTO 'parallel_run' ('effective_date', '""" + run + """') VALUES ('""" + str(
                    i) + """','""" + str(v) + """');"""
                cursor.execute(sqlite_insert_query)
            sqliteConnection.commit()
            cursor.close()
        except sqlite3.Error as error:
            print("Error while working with SQLite", error)
            success = False
        finally:
            if (sqliteConnection):
                sqliteConnection.close()
                print("The Sqlite connection is closed")
        return success

    def updateResults(self, run, values):
        sqliteConnection = None
        try:
            sqliteConnection = sqlite3.connect(self.db_path)
            cursor = sqliteConnection.cursor()
            for i, v in values.iteritems():
                print("Connected to SQLite")
                sqlite_update_query = """UPDATE 'parallel_run' set '""" + run + """' = '""" + str(
                    v) + """' where effective_date = '""" + str(i) + """'"""
                cursor.execute(sqlite_update_query)
            sqliteConnection.commit()
            cursor.close()
        except sqlite3.Error as error:
            print("Error while working with SQLite", error)
        finally:
            if (sqliteConnection):
                sqliteConnection.close()
                print("The Sqlite connection is closed")


    def addRunWeights(self,run, weights_df):
        sqliteConnection = None
        try:
            table_name = run + '_weight'
            sqliteConnection = sqlite3.connect(self.db_path)
            weights_df.to_sql(name=table_name, con=sqliteConnection)

        except sqlite3.Error as error:
            print("Error while creating a sqlite table", error)
        finally:
            if (sqliteConnection):
                sqliteConnection.close()
                print("sqlite connection is closed")

    def addResults(self, run, values):
        success = self.insertResults(run,values)
        if not success :
            self.updateResults(run, values)


    def saveResults(self, run, values, weights_df):
        self.addRun(run)
        self.addResults(run, values)
        self.totalRow = self.datesCount()
        self.addRunWeights(run, weights_df)

    def datesCount(self):
        sqliteConnection = None
        try:
            sqliteConnection = sqlite3.connect(self.db_path, timeout=20)
            cursor = sqliteConnection.cursor()
            print("Connected to SQLite")
            sqlite_select_query = """SELECT count(*) from parallel_run"""
            cursor.execute(sqlite_select_query)
            totalRows = cursor.fetchone()
            print("Total rows are:  ", totalRows)
            cursor.close()
            return totalRows
        except sqlite3.Error as error:
            print("Failed to read data from sqlite table", error)
        finally:
            if (sqliteConnection):
                sqliteConnection.close()
                print("The Sqlite connection is closed")


    def getAllRuns(self):
        sqliteConnection = None
        try:
            sqliteConnection = sqlite3.connect(self.db_path, timeout=20)
            cursor = sqliteConnection.cursor()
            print("Connected to SQLite")
            sqlite_select_runs =  """PRAGMA table_info(parallel_run);"""
            cursor.execute(sqlite_select_runs)
            all_runs_tuple = cursor.fetchall()

            cursor.close()
            all_runs = [r[1] for r in all_runs_tuple if not 'date' in r[1]]
            print("Total run numbers :  ", len(all_runs))

            return all_runs
        except sqlite3.Error as error:
            print("Failed to read data from sqlite table", error)
        finally:
            if (sqliteConnection):
                sqliteConnection.close()
                print("The Sqlite connection is closed")

    def analyzeRunResults(self, run):
        runs = self.getAllRuns()
        results_df = None

        if run not in runs :
            return results_df

        sqliteConnection = None

        try:
            sqliteConnection = sqlite3.connect(self.db_path, timeout=20)
            sqlite_select_run_query = """SELECT effective_date, """+run+""" from parallel_run order by effective_date asc"""
            results_df = pd.read_sql_query(sqlite_select_run_query, sqliteConnection)
        except sqlite3.Error as error:
            print("Failed to read data from sqlite table", error)
        finally:
            if (sqliteConnection):
                sqliteConnection.close()
                print("The Sqlite connection is closed")
        return results_df

    def analyzeAllRunResults(self):
        runs = self.getAllRuns()
        if len(runs)>1:
            runs = ','.join(runs)
        else :
            runs = runs[0]

        sqliteConnection = None

        try:
            sqliteConnection = sqlite3.connect(self.db_path, timeout=20)
            sqlite_select_run_query = """SELECT effective_date, """+runs+""" from parallel_run order by effective_date asc"""
            results_df = pd.read_sql_query(sqlite_select_run_query, sqliteConnection)
        except sqlite3.Error as error:
            print("Failed to read data from sqlite table", error)
        finally:
            if (sqliteConnection):
                sqliteConnection.close()
                print("The Sqlite connection is closed")


        results_df = results_df.rename(columns={"effective_date": "Date"})
        results_df['Date'] = pd.to_datetime(results_df['Date'])
        results_df = results_df.sort_values(by=['Date'])
        results_df = results_df.set_index(results_df['Date'])
        results_df = results_df.drop(['Date'], axis=1)


        run_empty_results = results_df.sum(axis = 0)
        self.empty_runs_to_investigate = list(run_empty_results.index[run_empty_results == 0])

        results_df = results_df.drop(columns=self.empty_runs_to_investigate)
        return results_df





