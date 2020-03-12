#! /usr/bin/python3
from datetime import datetime
import getpass
import pandas as pd  # untested on 1.+. Please use 0.25
import pyodbc  # Make sure to have an ODBC driver installed on your OS
#


class Querier:
    """ This class grabs data from the database and creates datatables
    for use in machine learning scripts.
    """

    def __init__(self, start, end, doe="experiments.csv"):
        """ Loads all the source data on the memory in a convenient
        form.

        For data originating from the machine:
            * Establishes connection with the database
            * Downloads the relevant experimental data
        The server address, username, and password are omitted. If
        you want to use this code yourself, replace the query stuff
        with your specific data loading processes.

        Separately, it reads in the experiment table for future
        joining. The experiment table has the records of all the
        experimental settings.
        """
        # Establish server connection:
        server_address = input("Server Address:")
        database = input("Database:")
        uid = input("Username:")
        password = getpass.getpass()
        connection_string = ("DRIVER={MySQL ODBC 8.0 Unicode Driver};"
                             + "SERVER=" + server_address
                             + ";DATABASE=" + database
                             + ";UID=" + uid
                             + ";PWD=" + password)
        connection = pyodbc.connect(connection_string)

        # Get data:
        query = ("SELECT Id, dateTimeReceived, payload FROM Messages " +
                 "WHERE Messages.dateTimeReceived >= '" + start +
                 "' AND Messages.dateTimeReceived <= '" + end + "'")
        self.results = pd.read_sql(query, connection)
        received_times = pd.to_datetime(self.results["dateTimeReceived"])
        self.results["dateTimeReceived"] = received_times.dt.tz_localize(None)

        # Load design of experiments table:
        self.doe = pd.read_csv(doe, header=0)
        numeric_cols = ["Set Cutting Depth (mm)",
                        "Set Finishing Feed Rate (mm/rev)",
                        "Set Surface Speed (m/min)"]
        for num_col in numeric_cols:
            self.doe[num_col] = pd.to_numeric(self.doe[num_col],
                                              errors="coerce")
        self.doe.rename(columns={"Tool": "Tool State"}, inplace=True)
        self.doe.reset_index(drop=True, inplace=True)

        # Display status:
        print("Downloaded data has " + str(len(self.results.index)) + " rows.")

    def save(self, file_name="results"):
        """ Saves data in various stages as csv.

        Use in mid-processing for debugging purposes.

        Use at the end fro the final data wrangling product.
        """
        self.results.to_csv(file_name+".csv", index=False)

    def split_g_code(self, line):
        """ Interprets the g code if present in the data reported by
        the machine.

        This is an optional step but it helps with
        some automatic labeling of the data as well as providing extra
        features for potential use in machine learning.
        """
        gcode, parameters, comment = None, None, None
        variable, value, = None, None
        test_or_exp, sample, repetition = None, None, None
        tool, coolant, cuttype, feed = None, None, None, None
        step, ffac, sfac = None, None, None
        index = ["G Code", "G Code Parameters", "Comment", "Variable",
                 "Variable Value", "Test or Experiment", "Sample",
                 "Repetition", "Tool", "Coolant", "Cut Type", "Feed",
                 "Cutting Depth (mm)", "Finishing Feed Rate (mm/rev)",
                 "Surface Speed (m/min)"]

        if type(line) is not str:
            return pd.Series([gcode, parameters, comment, variable, value,
                              test_or_exp, sample, repetition, tool, coolant,
                              cuttype, feed, step, ffac, sfac], index=index)
        if line == "":
            return pd.Series([gcode, parameters, comment, variable, value,
                              test_or_exp, sample, repetition, tool, coolant,
                              cuttype, feed, step, ffac, sfac], index=index)
        line = line.split("(", maxsplit=1)

        code = line[0]
        # Example experiment start message:
        # G96 S200 M3 (START SAMPLE 21 - STEP_0.50_FFAC_0.10_SFAC200)
        if len(line) > 1:
            comment = line[1].strip(" )")
            if comment.startswith("START SAMPLE"):
                test_or_exp = "Experiment"
                *_, sample = comment.split("-", maxsplit=1)[0].split()
                _, step, _, ffac, sfac = comment.split("_")
                step = float(step)
                ffac = float(ffac)
                sfac = float(sfac[4:-1])
            elif comment.startswith("STARTTEST"):
                test_or_exp = "Test"
                *_, sample = comment.split("-", maxsplit=1)[0].split()
            elif comment.startswith("STOP"):
                test_or_exp = "N/A"
                sample = "N/A"
                repetition = "N/A"

        code = code.strip().split("=", maxsplit=1)
        if len(code) == 1:
            gline = code[0].strip().split()
            gcode = gline[0]
            if len(gline) > 1:
                gparams = gline[1:]
            if gcode == "M8":
                coolant = 1
            elif gcode == "M9":
                coolant = 0
            elif gcode == "G1":
                if gparams[0][0] == "X":
                    cuttype = "Turning"
                elif gparams[0][0] == "Z":
                    cuttype = "Facing"
                feed_str = gparams[1][1:]
                if feed_str == "[FRGH]":
                    feed = "FRGH"  # will be replaced later
                else:
                    feed = feed_str
            elif gcode[0] == "T":
                tool = int(gcode[-1])
        else:
            variable = code[0].strip()
            value = float(code[1].strip())

        return pd.Series([gcode, parameters, comment, variable, value,
                          test_or_exp, sample, repetition, tool, coolant,
                          cuttype, feed, step, ffac, sfac], index=index)

    def process_sensor_data(self, sensor):
        """ Unpacks sensor data using information about sampling
        rate and current time.
        """
        sensor_rows = self.results[pd.notna(self.results[sensor])]
        array_col_num = sensor_rows.columns.get_loc(sensor)
        interval_col_num = sensor_rows.columns.get_loc("samplingInterval")
        time_col_num = sensor_rows.columns.get_loc("dateTime")
        id_col_num = sensor_rows.columns.get_loc("Id")
        received_col_num = sensor_rows.columns.get_loc("dateTimeReceived")
        asset_col_num = sensor_rows.columns.get_loc("assetId")
        sensor_col_num = sensor_rows.columns.get_loc("sensorId")
        if "itemInstanceId" in sensor_rows:
            instance_col_num = sensor_rows.columns.get_loc("itemInstanceId")
        curphase_col_num = sensor_rows.columns.get_loc("phase")

        array_dt_list = []  # this list grows but it's not a big problem
        for row in sensor_rows.itertuples():
            readings_list = row[array_col_num+1]
            this_dt = pd.DataFrame(readings_list, columns=[sensor])
            data_id = row[id_col_num+1]
            this_dt["Id"] = data_id
            this_dt["dateTimeReceived"] = row[received_col_num+1]
            this_dt["assetId"] = row[asset_col_num+1]
            this_dt["sensorId"] = row[sensor_col_num+1]
            if "itemInstanceId" in sensor_rows:
                this_dt["itemInstanceId"] = row[instance_col_num+1]
            this_dt["phase"] = row[curphase_col_num+1]
            if "samplingInterval" in sensor_rows:
                interval = row[interval_col_num+1]
            else:  # this shouldn't happen but sometime we get unexpected data
                interval = 0
            # tot_time = pd.Timedelta((len(readings_list) - 1) * interval, "S")
            start_time = pd.Timestamp(row[time_col_num+1])  # - tot_time
            interval = str(interval) + "S"
            if interval == "NANS":
                this_dt["dateTime"] = start_time
                print(data_id, interval)
            elif interval == "nanS":
                this_dt["dateTime"] = start_time
                print(data_id, interval)
            else:
                this_dt["dateTime"] = pd.date_range(start_time,
                                                    periods=len(readings_list),
                                                    freq=interval)
            array_dt_list.append(this_dt.copy())

        return pd.concat(array_dt_list, axis=0, ignore_index=True, sort=False)

    def parse_payloads(self):
        """ Parses data stored as strings of JSON payloads into
        columns.

        This is the most computationally difficult step. It could use
        some parallelization but this has not been implemented.
        """
        expanded_results = self.results["payload"].apply(pd.read_json,
                                                         typ="series")
        self.results = pd.concat([self.results, expanded_results], axis=1,
                                 ignore_index=False)

        """ Some payloads have "value", some have "values".
        This depends on whether the data is a single value vs. array.
        We do not need this distinction in the final table as arrays
        will be rolled out.
        """
        if "values" in self.results.columns.values.tolist():
            self.results["value"].update(self.results["values"])
            self.results.drop(columns=["values"], inplace=True)

        """ Payloads are already parsed.
        The original column isn't needed anymore
        """
        self.results.drop(columns=["payload"], inplace=True)

        # Formatting fixes
        times = pd.to_datetime(self.results["dateTime"])
        self.results["dateTime"] = times.dt.tz_localize(None)
        intervals = pd.to_numeric(self.results["samplingInterval"],
                                  errors="coerce")
        self.results["samplingInterval"] = intervals.round(6)
        clean_values = self.results["value"].replace({"UNAVAILABLE": None})
        self.results["value"] = clean_values
        self.results.sort_values(by=["dateTime", "Id"], axis=0, ascending=True,
                                 inplace=True)
        self.results.reset_index(drop=True, inplace=True)

        # Display progress:
        print("Payload parsing is finished.")

    def unstack_features(self):
        """ Separates features to columns and compresses the data table
        vertically.
        """
        pivoted = pd.pivot_table(self.results, values="value",
                                 columns="dataItemId",
                                 index=self.results.index, aggfunc="last")
        self.results = pd.concat([self.results, pivoted], axis=1,
                                 ignore_index=False)
        self.results.drop(columns=["dataItemId", "value"], inplace=True)
        self.results.sort_values(by=["dateTime", "Id"], axis=0, ascending=True,
                                 inplace=True)
        self.results.reset_index(drop=True, inplace=True)

        # Formatting fixes:
        if "LS1cmd" in self.results.columns.values.tolist():
            self.results["LS1cmd"] = pd.to_numeric(self.results["LS1cmd"],
                                                   errors="coerce")
        if "LS1load" in self.results.columns.values.tolist():
            self.results["LS1load"] = pd.to_numeric(self.results["LS1load"],
                                                    errors="coerce")
        if "LS1ovr" in self.results.columns.values.tolist():
            self.results["LS1ovr"] = pd.to_numeric(self.results["LS1ovr"],
                                                   errors="coerce")
        if "LS1speed" in self.results.columns.values.tolist():
            self.results["LS1speed"] = pd.to_numeric(self.results["LS1speed"],
                                                     errors="coerce")
        if "LX1actw" in self.results.columns.values.tolist():
            self.results["LX1actw"] = pd.to_numeric(self.results["LX1actw"],
                                                    errors="coerce")
        if "LX1load" in self.results.columns.values.tolist():
            self.results["LX1load"] = pd.to_numeric(self.results["LX1load"],
                                                    errors="coerce")
        if "LZ1actw" in self.results.columns.values.tolist():
            self.results["LZ1actw"] = pd.to_numeric(self.results["LZ1actw"],
                                                    errors="coerce")
        if "LZ1load" in self.results.columns.values.tolist():
            self.results["LZ1load"] = pd.to_numeric(self.results["LZ1load"],
                                                    errors="coerce")
        if "Lp1BlockNumber" in self.results.columns.values.tolist():
            block_number = self.results["Lp1BlockNumber"]
            self.results["Lp1BlockNumber"] = pd.to_numeric(block_number,
                                                           errors="coerce")
        if "Lp1CurrentTool" in self.results.columns.values.tolist():
            current_tool = self.results["Lp1CurrentTool"]
            self.results["Lp1CurrentTool"] = pd.to_numeric(current_tool,
                                                           errors="coerce")
        if "LpCuttingTime" in self.results.columns.values.tolist():
            cutting_time = self.results["LpCuttingTime"]
            self.results["LpCuttingTime"] = pd.to_numeric(cutting_time,
                                                          errors="coerce")
        if "LpFovr" in self.results.columns.values.tolist():
            self.results["LpFovr"] = pd.to_numeric(self.results["LpFovr"],
                                                   errors="coerce")
        if "LpRunningTime" in self.results.columns.values.tolist():
            running_time = self.results["LpRunningTime"]
            self.results["LpRunningTime"] = pd.to_numeric(running_time,
                                                          errors="coerce")
        if "LpSpindleRunTime" in self.results.columns.values.tolist():
            spindle_run_time = self.results["LpSpindleRunTime"]
            self.results["LpSpindleRunTime"] = pd.to_numeric(spindle_run_time,
                                                             errors="coerce")

        # Display progress:
        print("dataItemId pivot is finished.")

    def expand_features(self):
        """ Splits features that have multiple pieces of information.
        This step needs previous knowledge of the types of data
        reported by the machine. Identification is manual.
        """
        self.results = self.results.groupby("dateTime", sort=False,
                                            as_index=False).last()
        self.results.sort_values(by=["dateTime", "Id"], axis=0, ascending=True,
                                 inplace=True)
        self.results.reset_index(drop=True, inplace=True)

        concat_list = [self.results]
        try:
            xyz = self.results["Lp1LPathPos"].str.split(" ", expand=True)
            xyz.rename(columns={0: "x", 1: "y", 2: "z"}, inplace=True)
            xyz["x"] = pd.to_numeric(xyz["x"], errors="coerce")
            xyz["y"] = pd.to_numeric(xyz["y"], errors="coerce")
            xyz["z"] = pd.to_numeric(xyz["z"], errors="coerce")
            concat_list.append(xyz)
        except:  # bad practice, fix later
            print("Lp1LPathPos was not communicated by the machine.")

        gcode_df = self.results["Lp1block"].apply(self.split_g_code)
        if "Variable Value" in gcode_df.columns.values.tolist():
            var_value = gcode_df["Variable Value"]
            gcode_df["Variable Value"] = pd.to_numeric(var_value,
                                                       errors="coerce")
        concat_list.append(gcode_df)

        self.results = pd.concat(concat_list, axis=1, ignore_index=False)
        self.results.drop(columns="Lp1block", inplace=True)

        # Display progress:
        print("Feature expansion is finished.")

    def split_array_data_into_rows(self, col_names):
        """ Creates many single valued rows from a single row with
        arrayed data. This helps with creating statistics flexibly,
        rather than relying on the specific packaging of data by the
        edge processor.
        """
        all_results = []
        for sensor_name in col_names:
            all_results.append(self.process_sensor_data(sensor_name))
        all_results = pd.concat(all_results, axis=0, ignore_index=True,
                                sort=False)
        self.results.drop(columns=col_names, inplace=True)
        self.results = self.results.append(all_results, ignore_index=True,
                                           sort=False)
        if "itemInstanceId" in self.results:
            self.results.drop(columns=["samplingInterval", "sensorId",
                                       "itemInstanceId"], inplace=True)
        else:
            self.results.drop(columns=["samplingInterval", "sensorId"],
                              inplace=True)

        # Fix the clock sync problem between the machine and the edge computer:
        self.results["dateTime"] = self.results.apply(
            lambda row: (row["dateTime"] + pd.Timedelta(hours=1, seconds=41)
                         if row["assetId"] == "OKUMA-AMPF1"
                         else row["dateTime"]),
            axis=1)
        self.results.sort_values(by=["dateTime", "Id"], axis=0, ascending=True,
                                 inplace=True)
        self.results = self.results.groupby("dateTime", sort=False,
                                            as_index=False).last()
        self.results.sort_values(by=["dateTime", "Id"], axis=0, ascending=True,
                                 inplace=True)
        self.results.reset_index(drop=True, inplace=True)

        # Display progress:
        print("Rows were extended using sensor arrayed data.")

    def fill_and_change_types(self):
        """ Forward-fills appropriate columns. Sensor columns are never
        forward-filled.
        """
        self.results.replace({"UNAVAILABLE": None, "": None}, inplace=True)
        for col in self.results.columns.values.tolist():
            if col == "Lp1block":
                continue
            if col[0] == "L":
                self.results[col] = self.results[col].fillna(method="ffill")
        if "x" in self.results:
            self.results["x"] = self.results["x"].fillna(method="ffill")
            self.results["y"] = self.results["y"].fillna(method="ffill")
            self.results["z"] = self.results["z"].fillna(method="ffill")
        if "LX1actw" in self.results:
            filled_x_actw = self.results["LX1actw"].fillna(method="ffill")
            self.results["LX1actw"] = filled_x_actw
        if "LY1actw" in self.results:
            filled_y_actw = self.results["LY1actw"].fillna(method="ffill")
            self.results["LY1actw"] = filled_y_actw
        if "LZ1actw" in self.results:
            filled_z_actw = self.results["LZ1actw"].fillna(method="ffill")
            self.results["LZ1actw"] = filled_z_actw
        test_or_exp = self.results["Test or Experiment"].fillna(method="ffill")
        self.results["Test or Experiment"] = test_or_exp
        filled_sample = self.results["Sample"].fillna(method="ffill")
        self.results["Sample"] = pd.to_numeric(filled_sample, errors="coerce")
        filled_rep = self.results["Repetition"].fillna(method="ffill")
        self.results["Repetition"] = filled_rep
        self.results["Tool"] = self.results["Tool"].fillna(method="ffill")
        filled_coolant = self.results["Coolant"].fillna(method="ffill")
        self.results["Coolant"] = filled_coolant
        filled_cut_type = self.results["Cut Type"].fillna(method="ffill")
        self.results["Cut Type"] = filled_cut_type
        filled_feed = self.results["Feed"].fillna(method="ffill")
        self.results["Feed"] = pd.to_numeric(filled_feed,
                                             errors="coerce").divide(2)

        self.results = self.results.merge(self.doe,
                                          how="left",
                                          left_on="Sample",
                                          right_on="Sample ID")
        self.results.dropna(axis=1, how="all", inplace=True)
        if "Comment" in self.results:
            self.results.drop(columns="Comment", inplace=True)
        if "Sample ID" in self.results:
            self.results.drop(columns="Sample ID", inplace=True)

        # Display progress:
        print("Necessary fills and type changes were performed.")


print(datetime.now())
QUERIER = Querier(start='2019-10-11 14:10:10',
                  end='2019-10-11 16:11:38',
                  doe="experiments.csv")
# QUERIER.save("downloaded")  # Use for debugging only
print(datetime.now())
QUERIER.parse_payloads()
# QUERIER.save("parsed")  # Use for debugging only
print(datetime.now())
QUERIER.unstack_features()
# QUERIER.save("unstacked")  # Use for debugging only
print(datetime.now())
QUERIER.expand_features()
# QUERIER.save("expanded")  # Use for debugging only
print(datetime.now())
QUERIER.split_array_data_into_rows(["Vibration", "Current", "Temperature"])
# QUERIER.save("extended")  # Use for debugging only
print(datetime.now())
QUERIER.fill_and_change_types()
QUERIER.save("filled")  # Final output
print(datetime.now())
