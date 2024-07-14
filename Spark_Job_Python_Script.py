# Libraries to import
from pyspark.sql.functions import *
from pyspark.sql.types import *

# Enable join reorder for optimization
spark.conf.set("spark.databricks.delta.joinReorder.enabled", "true")

# 1. Select the datasets
dataset = dbutils.fs.head("dbfs:/databricks-datasets/lending-club-loan-stats/lendingClubData.txt")
print("Dataset head:", dataset)

# 2. Read the dataset
lending_club_df = spark.read \
    .option("inferSchema", "true") \
    .option("header", "true") \
    .csv("dbfs:/databricks-datasets/lending-club-loan-stats/LoanStats_2018Q2.csv")

print("Number of partitions:", lending_club_df.rdd.getNumPartitions())

# 3. Create member_Id using SHA-2 DSA to generate unique member_id
subset = lending_club_df.select('emp_title', 'emp_length', 'home_ownership', 'annual_inc', 'verification_status', 'grade', 'term', 'zip_code')
LCD_1 = lending_club_df.withColumn("member_id", sha2(concat_ws("||", *subset.columns), 256))
print("Step 3: Successfully generated unique member IDs using SHA-2 DSA.")

# 4. Remove biased (duplicate) member_id records

# Create a temporary DataFrame with member_id and their counts
LCD_TEMP = LCD_1.groupBy("member_id").count().withColumnRenamed("count", "member_count").where(col("member_count") > 1)

# Collect the member_ids with more than one occurrence into a list
member_ids_to_exclude = [row.member_id for row in LCD_TEMP.select("member_id").collect()]

# Filter the original DataFrame to exclude those member_ids
LCD_2 = LCD_1.filter(~col("member_id").isin(member_ids_to_exclude))
print("Step 4: Successfully removed biased (duplicate) member_id records.")

# 5. Data Modelling
# 1. borrower_info_df data
borrower_info_df = LCD_2.select('member_id', 'emp_title', 'emp_length', 'home_ownership', 
                                'annual_inc', 'verification_status', 'zip_code', 'addr_state')

# 2. loan_details_df data
loan_details_df = LCD_2.select('member_id', 'loan_amnt', 'funded_amnt', 'funded_amnt_inv', 
                                'term', 'int_rate', 'installment', 'grade', 'sub_grade', 'issue_d', 
                                'loan_status', 'pymnt_plan', 'purpose', 'title')

# 3. Payment and Recovery Dataset
payment_recovery_df = LCD_2.select('member_id', 'total_pymnt', 'total_pymnt_inv', 
                                    'total_rec_prncp', 'total_rec_int', 'total_rec_late_fee', 
                                    'recoveries', 'collection_recovery_fee', 'last_pymnt_d', 
                                    'last_pymnt_amnt', 'next_pymnt_d')

# 4. Credit and Risk Dataset
credit_risk_df = LCD_2.select('member_id', 'dti', 'delinq_2yrs', 'earliest_cr_line', 
                                'inq_last_6mths', 'mths_since_last_delinq', 'mths_since_last_record', 
                                'open_acc', 'pub_rec', 'revol_bal', 'revol_util', 'total_acc', 
                                'initial_list_status', 'out_prncp', 'out_prncp_inv')

# 6. Write back the segregated datasets into staging layer
# Write the dataframes into the staging layer of DBFS
dbutils.fs.mkdirs("/lending_club_project/staging/")

borrower_info_df.repartition(1).write.format("parquet").mode("overwrite").option("header", "true").save("/lending_club_project/staging/borrower_info/")
loan_details_df.repartition(1).write.format("parquet").mode("overwrite").option("header", "true").save("/lending_club_project/staging/loan_details/")
payment_recovery_df.repartition(1).write.format("parquet").mode("overwrite").option("header", "true").save("/lending_club_project/staging/payment_recovery/")
credit_risk_df.repartition(1).write.format("parquet").mode("overwrite").option("header", "true").save("/lending_club_project/staging/credit_risk/")

print("Step 5: Successfully created data subsets for borrower info, loan details, payment recovery, and credit risk.")

# 1. Cleaning of borrower_details_datasets
# 1a. Define schema and assign column names
schema = 'member_id string, emp_title string, emp_length string, home_ownership string, annual_inc double, verification_status string, zip_code string, addr_state string'
loan_info_df = spark.read.schema(schema).parquet("dbfs:/lending_club_project/staging/borrower_info")

# 1b. Add ingestion date and country column default as USA
dfa1 = loan_info_df.withColumn("ingestion_date", current_timestamp()).withColumn("country", lit("USA"))

# 1c. Remove records where annual_inc is zero (they are not eligible for credit loan)
dfa2 = dfa1.where(col("annual_inc") > 0)

# 1d. Convert 'null' employee title into 'Others' and convert the case into sentence case

# Define a Python function to convert text to sentence case
def to_sentence_case(text):
    if text:
        return text.title()
    return None

# Register the function as a UDF
to_sentence_case_udf = udf(to_sentence_case, StringType())

dfa2 = dfa2.withColumn("emp_title", to_sentence_case_udf(dfa2["emp_title"]))
dfa3 = dfa2.fillna({"emp_title": "Others"})

# 1e. Convert employee length into integer and revert n/a into 0
# Clean `emp_length` column using regexp_replace to keep only numeric characters
dfa3_cleaned = dfa3.withColumn("emp_length", regexp_replace(col("emp_length"), "[^0-9]", ""))

# Calculate average of emp_length (assuming it's numeric)
avg_emp_length = dfa3_cleaned.select(avg(col("emp_length"))).collect()[0][0]

# Replace blank cells with the calculated average and handle type casting
dfa4 = dfa3_cleaned.withColumn("emp_length", when(col("emp_length") == "", avg_emp_length).otherwise(col("emp_length"))) \
                   .withColumn("emp_length", col("emp_length").cast("int")) \
                   .withColumnRenamed("emp_length", "emp_length_in_years") \
                   .fillna({"emp_length_in_years": 6})  # Assuming you want to fill any remaining nulls

# 1f. Verify any anomaly present in zip_code column and addr_state column
zip_veri = dfa4.filter(length(col("zip_code")) > 5).count()
print(f"Number of records with zip_code length greater than 5: {zip_veri}")

addr_state_df = dfa4.filter(length(col("addr_state")) > 2).count()
print(f"Number of records with addr_state length greater than 2: {addr_state_df}")

# 2. Cleaning of loan_details_datasets
# 2a. Define schema and assign column names
lend_info_df = spark.read.option("inferSchema", "true").parquet("/lending_club_project/staging/loan_details/")

# 2b. Convert term column into tenure_period_in_month and cast it into integer
dfb1 = lend_info_df.withColumn("term", regexp_replace(col("term"), "[^0-9]", "")).withColumn("term", col("term").cast("int")).withColumnRenamed("term", "tenure_period_in_month")

# 2c. Convert int_rate into interest_rate_in_percentage and cast it into double
dfb2 = dfb1.withColumn("int_rate", regexp_replace(col("int_rate"), "[^0-9.]", "").cast("double")).withColumnRenamed("int_rate", "interest_rate_in_percentage")

# 2d. Convert issue_d into issued_date in date format
dfb3 = dfb2.withColumn("issue_d", to_date(col("issue_d"), "MMM-yy")).withColumnRenamed("issue_d", "issued_date")

# 2e. Cast purpose column into sentence case
to_sent_case_udf = udf(to_sentence_case, StringType())
dfb4 = dfb3.withColumn("purpose", to_sent_case_udf(col("purpose")))

# 2f. Drop unnecessary columns
dfb5 = dfb4.drop("pymnt_plan", "title")

# 3. Cleaning of payment_recovery_datasets
# 3a. Define schema and assign column names
payment_recovery_df = spark.read.option("inferSchema", "true").parquet("/lending_club_project/staging/payment_recovery/")

# 3b. Cast last_pymnt_d and next_pymnt_d to date and rename columns
dfc1 = payment_recovery_df.withColumn("last_pymnt_d", to_date(col("last_pymnt_d"), "MMM-yy")).withColumnRenamed("last_pymnt_d", "last_payment_date")
dfc2 = dfc1.withColumn("next_pymnt_d", to_date(col("next_pymnt_d"), "MMM-yy")).withColumnRenamed("next_pymnt_d", "next_payment_date")

# 4. Cleaning of Credit risk dataset

# 4a. Define schema and assign column name
credit_risk_df = spark.read.option("inferSchema", "true").parquet("/lending_club_project/staging/credit_risk/")

# Details to know
# 1. Personal and Loan Information: member_id uniquely identifies borrowers.
# 2. Financial Ratios: dti shows debt burden relative to income.
# 3. Delinquency Metrics: delinq_2yrs and mths_since_last_delinq track past due payments and recency.
# 4. Credit History: earliest_cr_line and inq_last_6mths show the length of credit history and recent inquiries.
# 5. Public Records: mths_since_last_record and pub_rec indicate public derogatory records.
# 6. Current Accounts: open_acc and total_acc show current and total credit accounts.
# 7. Revolving Credit: revol_bal and revol_util measure revolving credit balance and usage.
# 8. Loan Status: initial_list_status provides the initial status of loan listings.
# 9. Principal Balances: out_prncp and out_prncp_inv show outstanding principal balances from borrower and investor perspectives.

# 4a. Filling Null values
from pyspark.sql.functions import mean, col, to_date, stddev, regexp_replace

dfd1 = credit_risk_df.fillna({
    'dti': credit_risk_df.select(mean(col('dti'))).collect()[0][0],  # Fill with mean
    'delinq_2yrs': 0,  # Fill with 0 if no delinquency
    'inq_last_6mths': 0,
    'mths_since_last_delinq': -1,  # Placeholder for missing values
    'mths_since_last_record': -1,
    'pub_rec': 0
})

# 4b. Conversion of date earliest_cr_line
dfd2 = dfd1.withColumn("earliest_cr_line", to_date(col("earliest_cr_line"), 'MM/yyyy'))

# 4c. Compare both mean and standard deviation are in line or not
# Results found closer value data seems to be in line

dti_mean = dfd2.select(mean(col('dti'))).collect()[0][0]
print("DTI Mean:", dti_mean)

dti_stddev = dfd2.select(stddev(col('dti'))).collect()[0][0]
print("DTI Standard Deviation:", dti_stddev)

# 4d. Drop duplicates
dfd3 = dfd2.dropDuplicates()

# 4e. Cast revol_util into double
dfd4 = dfd3.withColumn("revol_util", regexp_replace(col("revol_util"), "[^0-9.]", "")).withColumnRenamed("revol_util","revol_util_in_percentage").drop("mths_since_last_record").withColumn("revol_util_in_percentage", col("revol_util_in_percentage").cast("float"))

# 5. Ultra cleaned Credit risk dataset
# 5a. Exclude the missing value i.e -1 from mths_since_last_delinq
dfd5 = dfd4.where(col("mths_since_last_delinq") != -1)

# 6. Write back datasets into delta tables
dbutils.fs.mkdirs("dbfs:/lending_club_project/landing")

dfa4.write.format("delta") \
    .mode("overwrite") \
    .option("path", "dbfs:/lending_club_project/landing/customers_details_datasets.delta") \
    .saveAsTable("customers_details_datasets")

dfb5.write.format("delta") \
    .mode("overwrite") \
    .option("path", "dbfs:/lending_club_project/landing/loan_details_datasets.delta") \
    .saveAsTable("loan_details_datasets")

dfc2.write.format("delta") \
    .mode("overwrite") \
    .option("path", "dbfs:/lending_club_project/landing/payment_recovery_datasets.delta") \
    .saveAsTable("payment_recovery_datasets")

dfd4.write.format("delta") \
    .mode("overwrite") \
    .option("path", "dbfs:/lending_club_project/landing/credit_risk_dataset.delta") \
    .saveAsTable("credit_risk_dataset")

dfd5.write.format("delta") \
    .mode("overwrite") \
    .option("path", "dbfs:/lending_club_project/landing/ultra_clean_credit_risk_dataset.delta") \
    .saveAsTable("ultra_clean_credit_risk_dataset")

# 7. Read delta tables
lending_club_crunched_df = spark.sql("""
    SELECT 
        U.member_id,
        C.emp_title, 
        C.emp_length_in_years, 
        C.home_ownership, 
        C.annual_inc, 
        C.verification_status, 
        C.zip_code, 
        C.addr_state, 
        C.ingestion_date, 
        C.country,
        L.loan_amnt, 
        L.funded_amnt, 
        L.funded_amnt_inv, 
        L.tenure_period_in_month, 
        L.interest_rate_in_percentage, 
        L.installment, 
        L.grade, 
        L.sub_grade, 
        L.issued_date, 
        L.loan_status, 
        L.purpose,
        P.total_pymnt, 
        P.total_pymnt_inv, 
        P.total_rec_prncp, 
        P.total_rec_int, 
        P.total_rec_late_fee, 
        P.recoveries, 
        P.collection_recovery_fee, 
        P.last_payment_date, 
        P.last_pymnt_amnt, 
        P.next_payment_date,
        U.dti, 
        U.delinq_2yrs, 
        U.earliest_cr_line, 
        U.inq_last_6mths, 
        U.mths_since_last_delinq, 
        U.open_acc, 
        U.pub_rec, 
        U.revol_bal, 
        U.revol_util_in_percentage, 
        U.total_acc, 
        U.initial_list_status, 
        U.out_prncp, 
        U.out_prncp_inv
    FROM ultra_clean_credit_risk_dataset U
    LEFT JOIN customers_details_datasets C ON U.member_id = C.member_id
    LEFT JOIN loan_details_datasets L ON U.member_id = L.member_id
    LEFT JOIN payment_recovery_datasets P ON U.member_id = P.member_id
""")

# Finalised total cleaned records
lending_club_crunched_df.write.format("delta") \
    .mode("overwrite") \
    .option("path", "dbfs:/lending_club_project/landing/lending_club_crunched_dataset.delta") \
    .saveAsTable("lending_club_crunched_dataset")


# BUSINESS USE CASE
# 8. Calculating Credit Score

# To calculate a credit score using a traditional approach without machine learning, you can use a scoring model based on weighted factors. Each factor is assigned a weight based on its importance in determining creditworthiness. Here's how you can do it:

# 1. Determine the Factors: Identify the key factors that will influence the credit score.
# 2. Assign Weights: Assign weights to each factor based on its importance.
# 3. Calculate Scores: Normalize and calculate the score for each factor.
# 4. Compute Final Credit Score: Aggregate the scores based on their weights.

# Example Credit Score Calculation
# Let's use the following factors:

# Debt-to-Income Ratio (dti)
# Number of Delinquencies in the Last 2 Years (delinq_2yrs)
# Revolving Balance (revol_bal)
# Revolving Utilization (revol_util_in_percentage)
# Number of Open Accounts (open_acc)
# Total Accounts (total_acc)
# Outstanding Principal (out_prncp)
# Outstanding Principal (Investor) (out_prncp_inv)

DF1 = spark.read.format("delta").option("inferSchema", "true").load("dbfs:/lending_club_project/landing/lending_club_crunched_dataset.delta")

# 8b. Drop rows with missing values in critical columns
DF2 = DF1.dropna(subset=["member_id", "loan_amnt", "funded_amnt", "annual_inc", "dti", "delinq_2yrs", "revol_bal", "total_acc", "out_prncp", "out_prncp_inv"])

# 8c. Filter out rows with revol_util_in_percentage > 100
DF3 = DF2.filter(col("revol_util_in_percentage") <= 100)

# 8d. Impute missing values in non-critical columns with the median or a placeholder value
median_annual_inc = DF3.approxQuantile("annual_inc", [0.5], 0.0)[0]
DF4 = DF3.withColumn("annual_inc", when(col("annual_inc").isNull(), median_annual_inc).otherwise(col("annual_inc")))

# 8e. Normalize and calculate scores for each factor
DF4 = DF4.withColumn("income_score", when(col("annual_inc") > 100000, 100)
                                    .when(col("annual_inc") > 75000, 75)
                                    .when(col("annual_inc") > 50000, 50)
                                    .when(col("annual_inc") > 25000, 25)
                                    .otherwise(0))

DF4 = DF4.withColumn("dti_score", when(col("dti") < 10, 100)
                                  .when(col("dti") < 20, 75)
                                  .when(col("dti") < 30, 50)
                                  .otherwise(0))

DF4 = DF4.withColumn("delinq_score", when(col("delinq_2yrs") == 0, 100)
                                     .when(col("delinq_2yrs") == 1, 50)
                                     .otherwise(0))

DF4 = DF4.withColumn("revol_bal_score", when(col("revol_bal") < 5000, 100)
                                      .when(col("revol_bal") < 10000, 75)
                                      .when(col("revol_bal") < 20000, 50)
                                      .otherwise(0))

DF4 = DF4.withColumn("revol_util_score", when(col("revol_util_in_percentage") < 30, 100)
                                       .when(col("revol_util_in_percentage") < 50, 75)
                                       .when(col("revol_util_in_percentage") < 70, 50)
                                       .otherwise(0))

DF4 = DF4.withColumn("total_acc_score", when(col("total_acc") > 20, 100)
                                      .when(col("total_acc") > 10, 75)
                                      .when(col("total_acc") > 5, 50)
                                      .otherwise(0))

DF4 = DF4.withColumn("out_prncp_score", when(col("out_prncp") == 0, 100)
                                      .when(col("out_prncp") < 1000, 75)
                                      .when(col("out_prncp") < 5000, 50)
                                      .otherwise(0))

DF4 = DF4.withColumn("out_prncp_inv_score", when(col("out_prncp_inv") == 0, 100)
                                          .when(col("out_prncp_inv") < 1000, 75)
                                          .when(col("out_prncp_inv") < 5000, 50)
                                          .otherwise(0))


# 8f. Aggregate Scores
DF5 = DF4.withColumn("total_score", col("income_score") + col("dti_score") + col("delinq_score") + col("revol_bal_score") + col("revol_util_score") + col("total_acc_score") + col("out_prncp_score") + col("out_prncp_inv_score"))

# 8g. Normalize the Score
max_score = 800  # Maximum possible score based on our criteria
min_score = 300  # Minimum possible score to match typical credit score ranges
DF6 = DF5.withColumn("credit_score", (col("total_score") / max_score) * (850 - 300) + 300)
print("Step 7: Successfully calculated and normalized the credit score.")

# 8h. Grading the Credit Score
DF7 = DF6.withColumn("credit_grade", when(col("credit_score") >= 750, "very_good")
                                    .when(col("credit_score") >= 700, "good")
                                    .when(col("credit_score") >= 650, "medium")
                                    .when(col("credit_score") >= 600, "bad")
                                    .otherwise("very_bad"))

very_good_df = DF7.where(col("credit_grade") == 'very_good')
good_df = DF7.where(col("credit_grade") == 'good')
medium_df = DF7.where(col("credit_grade") == 'medium')
bad_df = DF7.where(col("credit_grade") == 'bad')
very_bad_df = DF7.where(col("credit_grade") == 'very_bad')
print("Step 8: Successfully graded the credit score into categories.")

# 9. Write back the graded finalised results
dbutils.fs.mkdirs("dbfs:/lending_club_project/reporting")

very_good_df.write.format("delta") \
    .mode("overwrite") \
    .option("path", "dbfs:/lending_club_project/reporting/very_good_score_dataset.delta") \
    .saveAsTable("very_good_score_dataset")

good_df.write.format("delta") \
    .mode("overwrite") \
    .option("path", "dbfs:/lending_club_project/reporting/good_score_dataset.delta") \
    .saveAsTable("good_score_dataset")

medium_df.write.format("delta") \
    .mode("overwrite") \
    .option("path", "dbfs:/lending_club_project/reporting/medium_score_dataset.delta") \
    .saveAsTable("medium_score_dataset")

bad_df.write.format("delta") \
    .mode("overwrite") \
    .option("path", "dbfs:/lending_club_project/reporting/bad_score_dataset.delta") \
    .saveAsTable("bad_score_dataset")

very_bad_df.write.format("delta") \
    .mode("overwrite") \
    .option("path", "dbfs:/lending_club_project/reporting/very_bad_score_dataset.delta") \
    .saveAsTable("very_bad_score_dataset")
print("Step 9: Successfully saved the final graded datasets to Delta tables.")
