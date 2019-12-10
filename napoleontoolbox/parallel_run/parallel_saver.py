#!/usr/bin/env python3
# coding: utf-8

import sqlite3
import pandas as pd
class ParallelSaver():

    def __init__(self, db_path):
        self.db_path = db_path
        self.runs = []
        self.totalRow = 0
        self.instantiateTable()

    def instantiateTable(self):
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
        sqliteConnection = sqlite3.connect(self.db_path)
        cursor = sqliteConnection.cursor()
        try:
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




    def addResults(self, run, values):
        success = self.insertResults(run,values)
        if not success :
            self.updateResults(run, values)


    def saveResults(self, run, values):
        self.addRun(run)
        self.addResults(run, values)
        self.totalRow = self.datesCount()

    def datesCount(self):
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

        return results_df





