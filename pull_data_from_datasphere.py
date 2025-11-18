import requests
import pandas as pd

def get_datasphere_data_odata(token_url, client_id, client_secret, odata_url):
    # 1. Obtain an access token
    token_data = {
        'grant_type': 'client_credentials'
    }
    # Client ID and secret are sent using Basic Auth
    token_response = requests.post(token_url, data=token_data, auth=(client_id, client_secret))
    token_response.raise_for_status()
    access_token = token_response.json()['access_token']

    # 2. Call the OData API
    headers = {
        'Authorization': f'Bearer {access_token}',
        'Accept': 'application/json'
    }
    # Append ?$format=json if not already in the URL
    if '$format=json' not in odata_url:
        odata_url += '?' if '?' not in odata_url else '&'
        odata_url += '$format=json'

    data_response = requests.get(odata_url, headers=headers)
    data_response.raise_for_status()
    data = data_response.json()

    # The actual data is in the 'value' field for OData v4
    if 'value' in data:
        df = pd.DataFrame(data['value'])
        return df
    else:
        print("No data found or unexpected OData response format.")
        return None

# Example Usage:
# Replace with your actual SAP Datasphere OAuth details and OData URL
TOKEN_URL = 'https://yash.authentication.us10.hana.ondemand.com/oauth/token'
CLIENT_ID = 'sb-c94b4ac9-387b-4cc2-88e9-6e160a16787b!b35030|client!b655'
CLIENT_SECRET = 'b2cb3d46-ef97-4d15-8312-61b56941a7d5$_KbY8i_ja6CUPu0fp6PV-M33D3skb2Ds4IaH309G0is='
ODATA_URL = 'https://yash.us10.hcs.cloud.sap/api/v1/datasphere/consumption/analytical/DWC_TRAINING_2022/New_Analytic_Model_Sales/New_Analytic_Model_Sales'

df_odata = get_datasphere_data_odata(TOKEN_URL, CLIENT_ID, CLIENT_SECRET, ODATA_URL)

if df_odata is not None:
    print(df_odata.head())
