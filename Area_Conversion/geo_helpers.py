from google.cloud import bigquery
import config
import pandas as pd
import os
#Select path to credentials

#Construct a BigQuery client object.

class geo_client(bigquery.Client):
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"]=config.GOOGLE_APPLICATION_CREDENTIALS
    
    def get_fips(self,geo_type = None,**kwargs):
        '''
        Returns an FIPS for a given type and name of US geograpy. 
        Examples: 
        get_fips('State',State = 'New Jersey') returns 34
        get_fips('County',State = 'New Jersey', County = 'Essex') returns 34013

        '''
        if not geo_type:
            raise ValueError('geo_type must be \'State\', or \'County\'')

        if geo_type == 'State':
            #Next fix: Right now this works through a query but will be faster with a built-in dictionary lookup. 
            query = """
                SELECT area_name, state_fips_code
                from `bigquery-public-data.census_utility.fips_codes_all` 
                WHERE area_name = @state
            """
            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("state", "STRING", kwargs['State']),
                        ]
            )
            return(self.query(query, job_config=job_config).to_dataframe().state_fips_code[0])

        if geo_type == 'County':
            #Next fix: Adding an Array lookup will make this faster when looking for a lot of counties at once. 
            try:
                state_FIPS = self.get_fips('State',State = kwargs['State'])
            except KeyError:
                print('When using get_fips() with a County must include the State the county is in')
                raise

            query = """
                SELECT area_name, state_fips_code, county_fips_code
                from `bigquery-public-data.census_utility.fips_codes_all` 
                WHERE area_name = @county AND state_fips_code = @state_fips
            """
            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("state_fips", "STRING", state_FIPS),
                    bigquery.ScalarQueryParameter("county", "STRING", kwargs['County'])
                        ]
            )
            return(self.query(query, job_config=job_config).to_dataframe().county_fips_code[0])

        raise ValueError('geo_type must be \'State\', or \'County\'')




