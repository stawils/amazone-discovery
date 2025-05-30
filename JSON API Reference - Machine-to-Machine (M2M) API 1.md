---
title: "JSON API Reference - Machine-to-Machine (M2M) API"
source: "https://m2m.cr.usgs.gov/api/docs/reference/#login-token"
author:
  - "[[USGS - U.S. Geological Survey]]"
published:
created: 2025-05-30
description: "Query and order satellite images, aerial photographs, and cartographic products through the U.S. Geological Survey"
tags:
  - "clippings"
---
## API Endpoints

| Endpoint | Description | Link |
| --- | --- | --- |
| data-owner | Returns details about the data owner | [Documentation](https://m2m.cr.usgs.gov/api/docs/reference/#data-owner) |
| dataset | Get results by ID or name | [Documentation](https://m2m.cr.usgs.gov/api/docs/reference/#dataset) |
| dataset-browse | Lists all available browse configurations for a dataset | [Documentation](https://m2m.cr.usgs.gov/api/docs/reference/#dataset-browse) |
| dataset-bulk-products | Lists all available products for a dataset - this does not gaurantee scene availability | [Documentation](https://m2m.cr.usgs.gov/api/docs/reference/#dataset-bulk-products) |
| dataset-catalogs | Returns a list of the available dataset catalogs (pre-grouped sets of datasets) | [Documentation](https://m2m.cr.usgs.gov/api/docs/reference/#dataset-catalogs) |
| dataset-categories | Dataset Category Search | [Documentation](https://m2m.cr.usgs.gov/api/docs/reference/#dataset-categories) |
| dataset-clear-customization | Dataset Clear Customization | [Documentation](https://m2m.cr.usgs.gov/api/docs/reference/#dataset-clear-customization) |
| dataset-coverage | Dataset Coverage | [Documentation](https://m2m.cr.usgs.gov/api/docs/reference/#dataset-coverage) |
| dataset-download-options | Lists all available products for a dataset - this does not gaurantee scene availability | [Documentation](https://m2m.cr.usgs.gov/api/docs/reference/#dataset-download-options) |
| dataset-file-groups | Dataset File Groups | [Documentation](https://m2m.cr.usgs.gov/api/docs/reference/#dataset-file-groups) |
| dataset-filters | Dataset Filters | [Documentation](https://m2m.cr.usgs.gov/api/docs/reference/#dataset-filters) |
| dataset-get-customization | Dataset Get Customization | [Documentation](https://m2m.cr.usgs.gov/api/docs/reference/#dataset-get-customization) |
| dataset-get-customizations | Dataset Get Customizations | [Documentation](https://m2m.cr.usgs.gov/api/docs/reference/#dataset-get-customizations) |
| dataset-messages | Returns any notices regarding the given datasets features | [Documentation](https://m2m.cr.usgs.gov/api/docs/reference/#dataset-messages) |
| dataset-metadata | Returns all metadata fields for a given dataset | [Documentation](https://m2m.cr.usgs.gov/api/docs/reference/#dataset-metadata) |
| dataset-order-products | Lists all available products for a dataset - this does not gaurantee scene availability | [Documentation](https://m2m.cr.usgs.gov/api/docs/reference/#dataset-order-products) |
| dataset-search | Dataset Search | [Documentation](https://m2m.cr.usgs.gov/api/docs/reference/#dataset-search) |
| dataset-set-customization | Dataset Set Customization | [Documentation](https://m2m.cr.usgs.gov/api/docs/reference/#dataset-set-customization) |
| dataset-set-customizations | Dataset Set Customizations | [Documentation](https://m2m.cr.usgs.gov/api/docs/reference/#dataset-set-customizations) |
| download-complete-proxied | Update downloaded file size and status for proxied downloads | [Documentation](https://m2m.cr.usgs.gov/api/docs/reference/#download-complete-proxied) |
| download-eula | Download EULA | [Documentation](https://m2m.cr.usgs.gov/api/docs/reference/#download-eula) |
| download-labels | Gets a list of unique download labels | [Documentation](https://m2m.cr.usgs.gov/api/docs/reference/#download-labels) |
| download-options | Download Options | [Documentation](https://m2m.cr.usgs.gov/api/docs/reference/#download-options) |
| download-order-load | Prepares a download order for procesing by moving the scenes into the queue for processing | [Documentation](https://m2m.cr.usgs.gov/api/docs/reference/#download-order-load) |
| download-order-remove | Removes an order from the download queue | [Documentation](https://m2m.cr.usgs.gov/api/docs/reference/#download-order-remove) |
| download-remove | Removes an item from the download queue | [Documentation](https://m2m.cr.usgs.gov/api/docs/reference/#download-remove) |
| download-request | Inserts the requested download into the download queue | [Documentation](https://m2m.cr.usgs.gov/api/docs/reference/#download-request) |
| download-retrieve | Returns all available and previously requests but not completed downloads | [Documentation](https://m2m.cr.usgs.gov/api/docs/reference/#download-retrieve) |
| download-search | Searches for downloads within the queue, regardless of status, that match the given label | [Documentation](https://m2m.cr.usgs.gov/api/docs/reference/#download-search) |
| download-summary | Gets a summary of all downloads, by dataset, for any matching labels | [Documentation](https://m2m.cr.usgs.gov/api/docs/reference/#download-summary) |
| grid2ll | Grid to Lat/Lng Conversion | [Documentation](https://m2m.cr.usgs.gov/api/docs/reference/#grid2ll) |
| login-app-guest | API Application Guest Login | [Documentation](https://m2m.cr.usgs.gov/api/docs/reference/#login-app-guest) |
| login-sso | API Login using ERS Single Sign-On (SSO) as a credential - the users ERS SSO Cookie must be available in this session | [Documentation](https://m2m.cr.usgs.gov/api/docs/reference/#login-sso) |
| login-token | API Login - Uses ERS Application Token | [Documentation](https://m2m.cr.usgs.gov/api/docs/reference/#login-token) |
| logout | API Logout | [Documentation](https://m2m.cr.usgs.gov/api/docs/reference/#logout) |
| notifications | Get system notifications | [Documentation](https://m2m.cr.usgs.gov/api/docs/reference/#notifications) |
| order-products | Gets the list of products for the given scenes | [Documentation](https://m2m.cr.usgs.gov/api/docs/reference/#order-products) |
| order-submit | Creates a new TRAM order | [Documentation](https://m2m.cr.usgs.gov/api/docs/reference/#order-submit) |
| permissions | Returns a list of user permissions | [Documentation](https://m2m.cr.usgs.gov/api/docs/reference/#permissions) |
| placename | Get results by search for a placename | [Documentation](https://m2m.cr.usgs.gov/api/docs/reference/#placename) |
| rate-limit-summary | Gets a summary of all downloads by status, and how close users are getting to the rate limit | [Documentation](https://m2m.cr.usgs.gov/api/docs/reference/#rate-limit-summary) |
| scene-list-add | Adds items in the given scene list | [Documentation](https://m2m.cr.usgs.gov/api/docs/reference/#scene-list-add) |
| scene-list-get | Returns items in the given scene list | [Documentation](https://m2m.cr.usgs.gov/api/docs/reference/#scene-list-get) |
| scene-list-remove | Rremoves items from the given list | [Documentation](https://m2m.cr.usgs.gov/api/docs/reference/#scene-list-remove) |
| scene-list-summary | Returns summary information for a given list | [Documentation](https://m2m.cr.usgs.gov/api/docs/reference/#scene-list-summary) |
| scene-list-types | Returns a list of list types available to the user | [Documentation](https://m2m.cr.usgs.gov/api/docs/reference/#scene-list-types) |
| scene-metadata | Scene Metadata | [Documentation](https://m2m.cr.usgs.gov/api/docs/reference/#scene-metadata) |
| scene-metadata-list | Scene Metadata where the input is a pre-set list | [Documentation](https://m2m.cr.usgs.gov/api/docs/reference/#scene-metadata-list) |
| scene-metadata-xml | Returns metadata formatted in XML, ahering to FGDC, ISO and EE scene metadata formatting standards | [Documentation](https://m2m.cr.usgs.gov/api/docs/reference/#scene-metadata-xml) |
| scene-search | Scene Search | [Documentation](https://m2m.cr.usgs.gov/api/docs/reference/#scene-search) |
| scene-search-delete | Scene Search - Deleted Scenes | [Documentation](https://m2m.cr.usgs.gov/api/docs/reference/#scene-search-delete) |
| scene-search-secondary | Scene Search - Secondary Dataset | [Documentation](https://m2m.cr.usgs.gov/api/docs/reference/#scene-search-secondary) |
| tram-order-detail-update | Updates the system details of a TRAM order | [Documentation](https://m2m.cr.usgs.gov/api/docs/reference/#tram-order-detail-update) |
| tram-order-details | Gets the system details of a TRAM order | [Documentation](https://m2m.cr.usgs.gov/api/docs/reference/#tram-order-details) |
| tram-order-details-clear | Clears the system details of a TRAM order | [Documentation](https://m2m.cr.usgs.gov/api/docs/reference/#tram-order-details-clear) |
| tram-order-details-remove | Removes a specific system details of a TRAM order | [Documentation](https://m2m.cr.usgs.gov/api/docs/reference/#tram-order-details-remove) |
| tram-order-search | Searches TRAM order | [Documentation](https://m2m.cr.usgs.gov/api/docs/reference/#tram-order-search) |
| tram-order-status | Gets the status of a TRAM order | [Documentation](https://m2m.cr.usgs.gov/api/docs/reference/#tram-order-status) |
| tram-order-units | Lists all units for a given TRAM order | [Documentation](https://m2m.cr.usgs.gov/api/docs/reference/#tram-order-units) |
| user-preference-get | Get User Preferences | [Documentation](https://m2m.cr.usgs.gov/api/docs/reference/#user-preference-get) |
| user-preference-set | Set User Preferences | [Documentation](https://m2m.cr.usgs.gov/api/docs/reference/#user-preference-set) |

#### data-owner

This method is used to provide the contact information of the data owner.

##### Input Parameters

| Parameter Name | Data Type | Required | Description |
| --- | --- | --- | --- |
| dataOwner | string | Yes | Used to identify the data owner - this value comes from the dataset-search response |
| Parameter Name | Data Type | Required | Description |

##### Examples

**Sample Request**  
```
{
    "dataOwner": "DMID"
}
```

**Sample Response**  
```
{
    "data": {
        "city": "Sioux Falls",
        "email": "custserv@usgs.gov",
        "phone": "1-800-252-4547",
        "address": "U.S. Geological Survey (USGS) Earth Resources Observation and Science (EROS) Center 47914 252nd Street",
        "country": "USA",
        "postalCode": "57198-0001",
        "contactName": "Customer Service Representative",
        "organizationName": "U.S. Geological Survey (USGS) Earth Resources Observation and Sc"
    },
    "version": "stable",
    "errorCode": null,
    "requestId": 6604,
    "sessionId": 1085,
    "errorMessage": ""
}
```

#### dataset

This method is used to retrieve the dataset by id or name.

##### Input Parameters

| Parameter Name | Data Type | Required | Description |
| --- | --- | --- | --- |
| datasetId | string | No | The dataset identifier - must use this or datasetName |
| datasetName | string | No | The system-friendly dataset name - must use this or datasetId |
| Parameter Name | Data Type | Required | Description |

##### Examples

**Sample Request**  
```
{
    "datasetName": "gls_all"
}
```

**Sample Response**  
```
{
    "data": {
        "catalogs": [
            "EE",
            "GV"
        ],
        "keywords": "Landsat 7 Enhanced Thematic Mapper Plus (ETM+),Global Land Survey (GLS),Landsat,Satellite Imagery,Remotely Sensed Imagery,Landsat 5 Thematic Mapper (TM),Visible Wavelengths,Thermal Wavelengths,Infrared Wavelengths",
        "legacyId": null,
        "dataOwner": "DMID",
        "datasetId": "5e7c4182eba11e53",
        "doiNumber": "https:\/\/doi.org\/10.5066\/F7M32TQB",
        "sceneCount": 42990,
        "dateUpdated": "2020-03-26 00:45:38.587013-05",
        "abstractText": "The Global Land Survey (GLS) datasets are a collection of orthorectified, cloud-minimized Landsat-type satellite images, providing near complete coverage of the global land area decadally since the early 1970s.  The global mosaics are centered on 1975, 1990, 2000, 2005, and 2010, and consist of data acquired from five sensors: Operational Land Imager, Enhanced Thematic Mapper Plus, Thematic Mapper, Multispectral Scanner, and Advanced Land Imager.  This newest version combines all of the GLS data into one collection which has all of the combined collections.  The GLS datasets have been widely used in land-cover and land-use change studies at local, regional, and global scales.  This study evaluates the GLS datasets with respect to their spatial coverage, temporal consistency, geodetic accuracy, radiometric calibration consistency, image completeness, extent of cloud contamination, and residual gaps.  The datasets have been improved in order to give spatial continuity across all decadal collections.  Most of the imagery (85%) having cloud cover of less than 10%, the acquisition years clustered much more tightly around their target years, better co-registration relative to GLS-2000, and better radiometric absolute calibration.  Probably, the most significant impediment to scientific use of the datasets is the variability of image phenology (i.e., acquisition day of year).  This collection provides end-users with an assessment of the quality of the GLS datasets for specific applications, and where possible, suggestions for mitigating their deficiencies.",
        "datasetAlias": "gls_all",
        "spatialBounds": {
            "east": 180.01,
            "west": -180.01,
            "north": 82.6855268254304,
            "south": -69.3502337564158
        },
        "acquisitionEnd": "2012-01-30",
        "collectionName": "Global Land Survey",
        "ingestFrequency": null,
        "acquisitionStart": "1972-07-25",
        "temporalCoverage": "[\"1972-07-25 00:00:00-05\",\"2012-01-30 00:00:00-06\"]",
        "supportCloudCover": true,
        "collectionLongName": "Global Land Survey",
        "datasetCategoryName": "Landsat Legacy",
        "supportDeletionSearch": false
    },
    "version": "stable",
    "errorCode": null,
    "requestId": 34465,
    "sessionId": null,
    "errorMessage": ""
}
```

#### dataset-browse

This request is used to return the browse configurations for the specified dataset.

##### Input Parameters

| Parameter Name | Data Type | Required | Description |
| --- | --- | --- | --- |
| datasetId | string | Yes | Determines which dataset to return browse configurations for |
| Parameter Name | Data Type | Required | Description |

##### Examples

**Sample Request**  
```
{
    "datasetId": "5e81f14f59432a27"
}
```

**Sample Response**  
```
{
    "requestId": 123456,
    "version": "stable",
    "data": [
        {
            "id": "5e81f7d39594374",
            "browseName": "Reflective Browse - Bands 6, 5, 4",
            "browseSource": "ls_chs",
            "browseSourceName": "Landsat CHS",
            "browseRotationEnabled": false,
            "browseKmzEnabled": true,
            "isGeolocated": true,
            "displayOrder": 3,
            "overlaySpec": "wmst"
        },
        ...
    ],
    "errorCode": null,
    "errorMessage": null
}
```

#### dataset-bulk-products

Lists all available bulk products for a dataset - this does not guarantee scene availability.

##### Input Parameters

| Parameter Name | Data Type | Required | Description |
| --- | --- | --- | --- |
| datasetName | string | No | Used to identify the which dataset to return results for |
| Parameter Name | Data Type | Required | Description |

##### Examples

**Sample Request**  
```
{
    "datasetName": "gls_all"
}
```

**Sample Response**  
```
{
    "data": [
        {
            "productCode": "D623",
            "productName": "Full Resolution Thermal Browse",
            "downloadName": null,
            "fileGroups": null
        },
        {
            "productCode": "D624",
            "productName": "Full Resolution Browse Bundle",
            "downloadName": null,
            "fileGroups": null
        },
        {
            "productCode": "D621",
            "productName": "GeoTIFF",
            "downloadName": null,
            "fileGroups": null
        },
        {
            "productCode": "D622",
            "productName": "Full Resolution Reflective Browse",
            "downloadName": null,
            "fileGroups": null
        }
    ],
    "version": "stable",
    "errorCode": null,
    "requestId": 11078,
    "sessionId": 1479,
    "errorMessage": ""
}
```

#### dataset-catalogs

This method is used to retrieve the available dataset catalogs. The use of dataset catalogs are not required, but are used to group datasets by their use within our web applications.

##### Input Parameters

<table><caption>Displays dataset-catalogs Input Parameters</caption><thead><tr><th>Parameter Name</th><th>Data Type</th><th>Required</th><th>Description</th></tr></thead><tbody><tr><td colspan="4">No Parameters for Endpoint</td></tr></tbody><tfoot><tr><td>Parameter Name</td><td>Data Type</td><td>Required</td><td>Description</td></tr></tfoot></table>

##### Examples

**Sample Response**  
```
{
    "data": {
        "EE": "EarthExplorer",
        "GV": "GloVis",
        "HDDS": "HDDS Explorer",
    },
    "version": "stable",
    "errorCode": null,
    "requestId": 34465,
    "sessionId": null,
    "errorMessage": ""
}
```

#### dataset-categories

This method is used to search datasets under the categories.

##### Input Parameters

| Parameter Name | Data Type | Required | Description |
| --- | --- | --- | --- |
| catalog | string | No | Used to identify datasets that are associated with a given application |
| includeMessages | boolean | No | Optional parameter to include messages regarding specific dataset components |
| publicOnly | boolean | No | Used as a filter out datasets that are not accessible to unauthenticated general public users |
| useCustomization | boolean | No | Used as a filter out datasets that are excluded by user customization |
| parentId | string | No | If provided, returned categories are limited to categories that are children of the provided ID |
| datasetFilter | string | No | If provided, filters the datasets - this automatically adds a wildcard before and after the input value |
| Parameter Name | Data Type | Required | Description |

##### Examples

**Sample Request**  
```
{
    "catalog": "EE",
    "publicOnly": false,
    "useCustomization": false
}
```

**Sample Response**  
```
{
    "data": {
        "2": {
            "id": "41",
            "datasets": [
                {
                    "catalogs": [
                        "EE"
                    ],
                    "keywords": "None,Non USGS,Indexes,Johnson Space Center (JSC),Bureau of Land Management (BLM),Ames,Photos,Bureau of Reclamation (BOR),Photography,Aerial,Earth Resources Observation and Science (EROS),Photo Mosaic,Imagery,National Park Service (NPS),Reconnaissance,EROS Data Center (EDC)",
                    "dataOwner": "DMID",
                    "datasetId": "5e83d8c98ce09e2",
                    "doiNumber": "https:\/\/doi.org\/10.5066\/F72805WQ",
                    "sceneCount": 0,
                    "dateUpdated": "2020-03-31 18:56:58.010115-05",
                    "abstractText": "USGS and Non USGS Agencies Aerial Photo Reference Mosaics inventory contains indexes to aerial photographs. The inventory contains imagery from various government agencies that are now archived at the USGS Earth Resources Observation and Science (EROS) Center. The film types, scales, and acquisition schedules differed according to project requirements. Low-, middle-, and high-altitude photographs were collected.\r\n ",
                    "datasetAlias": "aerial_combin_index",
                    "spatialBounds": {
                        "east": 174.201,
                        "west": -170.901,
                        "north": 71.351,
                        "south": -14.401
                    },
                    "acquisitionEnd": "1989-03-22",
                    "collectionName": "Aerial Photo Mosaics",
                    "ingestFrequency": null,
                    "acquisitionStart": "1937-10-24",
                    "temporalCoverage": null,
                    "supportCloudCover": true,
                    "collectionLongName": "Aerial Photo Mosaics = Photo Indexes and Map-Line Plots: Pre 1990",
                    "datasetCategoryName": "Aerial Imagery",
                    "supportDeletionSearch": false
                }
            ],
            "categoryName": "ISRO Resourcesat",
            "referenceLink": null,
            "subCategories": [],
            "parentCategoryId": null,
            "parentCategoryName": null,
            "categoryDescription": null
        }
    },
    "version": "stable",
    "errorCode": null,
    "requestId": 27662,
    "sessionId": 1929,
    "errorMessage": ""
}
```

#### dataset-clear-customization

This method is used the remove an entire customization or clear out a specific metadata type.

##### Input Parameters

| Parameter Name | Data Type | Required | Description |
| --- | --- | --- | --- |
| datasetName | string | No | Used to identify the dataset to clear. If null, all dataset customizations will be cleared. |
| metadataType | string\[\] | No | If populated, identifies which metadata to clear(export, full, res\_sum, shp) |
| fileGroupIds | string\[\] | No | If populated, identifies which file group to clear |
| Parameter Name | Data Type | Required | Description |

##### Examples

**Sample Request**  
```
{ 
    "datasetName": "landsat_band_files_c2_l1",
    "metadataType": ["shp", "full"],
    "fileGroupIds": ["ls_c2l1_all"]
}
```

**Sample Response**  
```
{
    "data": 1,
    "version": "stable",
    "errorCode": null,
    "requestId": 1075864,
    "sessionId": null,
    "errorMessage": ""
}
```

#### dataset-coverage

Returns coverage for a given dataset.

##### Input Parameters

| Parameter Name | Data Type | Required | Description |
| --- | --- | --- | --- |
| datasetName | string | Yes | Determines which dataset to return coverage for |
| Parameter Name | Data Type | Required | Description |

##### Examples

**Sample Request**  
```
{
    "datasetName": "gls_all"
}
```

**Sample Response**  
```
{
    "data": {
        "bounds": {
            "type": "Polygon",
            "coordinates": [
                [
                    [
                        -180.01,
                        -69.3502337564158
                    ],
                    [
                        -180.01,
                        82.6855268254304
                    ],
                    ...
                ] 
            ] 
        }, 
        "geoJson": {
            "type": "MultiPolygon", 
            "coordinates": [ 
                [ 
                    [ 
                        [ 
                            -91.575245613424, 
                            -69.1666591427406 
                        ], 
                        [ 
                            -91.4884509363067, 
                            -69.0939312836925 
                        ],
                        ... 
                    ]
                ]
            ]
        }
        ...
    },
    "version": "stable",
    "errorCode": null,
    "requestId": 28043,
    "sessionId": null,
    "errorMessage": ""
```

#### dataset-download-options

This request lists all available products for a given dataset - this does not guarantee scene availability.

##### Input Parameters

| Parameter Name | Data Type | Required | Description |
| --- | --- | --- | --- |
| datasetName | string | Yes | Used to identify the which dataset to return results for |
| sceneFilter | [SceneFilter](https://m2m.cr.usgs.gov/api/docs/datatypes/#sceneFilter) | No | Used to filter data within the dataset |
| Parameter Name | Data Type | Required | Description |

##### Examples

**Sample Request**  
```
{
    "datasetName": "gls_all",
    "sceneFilter": {
        "spatialFilter": {
            "filterType": "mbr",
            "lowerLeft": {
                    "latitude": 44.60847,
                    "longitude": -99.69639
            },
            "upperRight": {
                    "latitude": 44.60847,
                    "longitude": -99.69639
            }
        },
        "metadataFilter": null,
        "cloudCoverFilter": {
            "max": 100,
            "min": 0,
            "includeUnknown": True
        },
        "acquisitionFilter": null
    },
}
```

**Sample Response**  
```
{
    "data": [
        {
            "productId": "5e7c4182c28455fb",
            "productCode": "D622",
            "productName": "Full Resolution Reflective Browse",
            "downloadName": null,
            "downloadSystem": "wms",
            "fileGroups": null
        },
        ...
    ],
    "version": "stable",
    "errorCode": null,
    "requestId": 12345678,
    "sessionId": 123456,
    "errorMessage": ""
}
If available checksum_types will be included in the response to show potential file validation options.
Sample Format:
"checksum_types": [
                {
                    "id": "cksum"
                },
                {
                    "id": "sha256sum"
                }
            ]
```

#### dataset-file-groups

This method is used to list all configured file groups for a dataset.

##### Input Parameters

| Parameter Name | Data Type | Required | Description |
| --- | --- | --- | --- |
| datasetName | string | Yes | Dataset alias |
| Parameter Name | Data Type | Required | Description |

##### Examples

**Sample Request**  
```
{ 
    "datasetName": "landsat_ot_c2_l1"
}
```

**Sample Response**  
```
{
    "requestId": 123456,
    "version": "stable",
    "data": {
        "primary": [],
        "secondary": {
            "5e81ff5f86d11d08": {
                "ls_c2l1_all": {
                    "id": "ls_c2l1_all",
                    "color": null,
                    "description": "Landsat Collection-2 Level-1 All Files",
                    "displayOrder": 0,
                    "icon": null,
                    "label": "All Level-1 Files",
                    "files": [
                        {
                            "name": "ANG.txt",
                            "productIds": {
                                "rp9zq1ewx8s985mp": "ls_s3",
                                "gy7bvmukhtkl910o": "dds"
                            },
                            "productName": "Landsat Collection 2 Level-1 Band File",
                            "displayOrder": 5
                        },                       
                        ...
                    ]
                }
            }
        }
    },
    "errorCode": null,
    "errorMessage": null
}
```

#### dataset-filters

This request is used to return the metadata filter fields for the specified dataset. These values can be used as additional criteria when submitting search and hit queries.

##### Input Parameters

| Parameter Name | Data Type | Required | Description |
| --- | --- | --- | --- |
| datasetName | string | Yes | Determines which dataset to return filters for |
| Parameter Name | Data Type | Required | Description |

##### Examples

**Sample Request**  
```
{
    "datasetName": "GLS_ALL"
}
```

**Sample Response**  
```
{
    "data": [
        {
            "id": "5e7c41828d69edad",
            "searchSql": "ENTITY_ID = ?",
            "fieldLabel": "Entity ID",
            "fieldConfig": {
                "type": "Text",
                "filters": [
                    {
                        "type": "StringToUpper",
                        "options": []
                    },
                    {
                        "type": "Application\\Filter\\Like",
                        "options": []
                    }
                ],
                "options": {
                    "size": "35"
                },
                "validators": [],
                "numElements": "5",
                "displayListId": null
            },
            "legacyFieldId": 21366,
            "dictionaryLink": https:\/\/www.usgs.gov\/centers\/eros\/science\/global-land-survey-data-dictionary?qt-science_center_objects=0#entity_id"
        },
        ...
    ],
    "version": "stable",
    "errorCode": null,
    "requestId": 26196,
    "sessionId": null,
    "errorMessage": ""
}
```

#### dataset-get-customization

This method is used to retrieve metadata customization for a specific dataset.

##### Input Parameters

| Parameter Name | Data Type | Required | Description |
| --- | --- | --- | --- |
| datasetName | string | Yes | Used to identify the dataset to search |
| Parameter Name | Data Type | Required | Description |

##### Examples

**Sample Request**  
```
{
    "datasetName": "landsat_band_files"
}
```

**Sample Response**  
```
{
    "data": {
        "excluded": false,
        "metadata": {
            "full": [
                {
                    "id": "5e839f1361280c84",
                    "sortOrder": 1
                },
                {
                    "id": "5e839f13baf6806e",
                    "sortOrder": 2
                },
                { ... }
            ],
            "export": [
                { ... },              
            ],
            "res_sum": [
                { ... },
            ]
        },
        "search_sort": [],
        "file_groups": {
            "ls_c2l1_ot_band": [
                "63ceb05468b7e8b4"
            ]
        }
    },
    "version": "stable",
    "errorCode": null,
    "requestId": 1075867,
    "sessionId": null,
    "errorMessage": ""
}
```

#### dataset-get-customizations

This method is used to retrieve metadata customizations for multiple datasets at once.

##### Input Parameters

| Parameter Name | Data Type | Required | Description |
| --- | --- | --- | --- |
| datasetNames | string\[\] | No | Used to identify the dataset(s) to return. If null it will return all the users customizations |
| metadataType | string\[\] | No | If populated, identifies which metadata to return(export, full, res\_sum, shp) |
| Parameter Name | Data Type | Required | Description |

##### Examples

**Sample Request**  
```
{
    "datasetNames": [
        "asas",
        "gls_all",
        "landsat_band_files_c2_l1"
    ]
}
```

**Sample Response**  
```
{
    "data": {
        "5e839f126ac633be": [
            {
                "excluded": false,
                "metadata": {
                    "full": [
                        {
                            "id": "5e839f1361280c84",
                            "sortOrder": null
                        },
                        { ... }
                    ],
                    "export": [
                        { ... }
                    ],
                    "res_sum": [
                        { ... }
                    ]
                },
                "search_sort": [],
                "file_groups": null
            }
        ],
        " ... ": [
            {
                "excluded": false,
                "metadata": {
                    "full": [
                        {
                            "id": "5ecf5a00365ae18e",
                            "sortOrder": 1
                        }
                    ],
                    "res_sum": [
                        { ... }
                    ]
                },
                "search_sort": [
                    {
                        "id": "5ecf5a00365ae18e",
                        "direction": "ASC",
                        "field_name": "Product Codr"
                    },
                    { ... }
                ],
                "file_groups": null
            }
        ]
    },
    "version": "stable",
    "errorCode": null,
    "requestId": 1077267,
    "sessionId": null,
    "errorMessage": ""
}
```

#### dataset-messages

Returns any notices regarding the given datasets features.

##### Input Parameters

| Parameter Name | Data Type | Required | Description |
| --- | --- | --- | --- |
| catalog | string | No | Used to identify datasets that are associated with a given application |
| datasetName | string | No | The system-friendly dataset name |
| datasetNames | string\[\] | No | Array of system-friendly dataset names |
| Parameter Name | Data Type | Required | Description |

##### Examples

**Sample Request**  
```
{
    "catalog": "EE",
    "datasetName": "GLS_ALL"
}
```

**Sample Response**  
```
{
    "data": {
        "gls_all": {
            "order": {
                "message": "test",
                "datasetName": "gls_all",
                "messageCode": "order"
            },
            "filter": {
                "message": "test",
                "datasetName": "gls_all",
                "messageCode": "filter"
            },
            "result": {
                "message": "test",
                "datasetName": "gls_all",
                "messageCode": "result"
            },
            "download": {
                "message": "test",
                "datasetName": "gls_all",
                "messageCode": "download"
            },
            "order-summary": {
                "message": "test",
                "datasetName": "gls_all",
                "messageCode": "order-summary"
            },
            "dataset-select": {
                "message": "test",
                "datasetName": "gls_all",
                "messageCode": "dataset-select"
            }
        }
    },
    "version": "stable",
    "errorCode": null,
    "requestId": 28045,
    "sessionId": null,
    "errorMessage": ""
}
```

This method is used to retrieve all metadata fields for a given dataset.

##### Input Parameters

| Parameter Name | Data Type | Required | Description |
| --- | --- | --- | --- |
| datasetName | string | Yes | The system-friendly dataset name |
| Parameter Name | Data Type | Required | Description |

##### Examples

**Sample Request**  
```
{
    "datasetName": "asas"
}
```

**Sample Response**  
```
{
    "data": {
        "shp": [
            {
                "id": "5e839f13a4606d54",
                "field_name": "ACQ_DATE",
                "display_order": 2,
                "result_set_type": "shp"
            },
            {
                "id": "5e839f13ffb41a9",
                "field_name": "PROJECT",
                "display_order": 3,
                "result_set_type": "shp"
            },
            { ... }
        ],
        "full": [
            { ... }
        ],
        "export": [
            { ... }
        ],
        "res_sum": [
            { ... }
        ]
    },
    "version": "stable",
    "errorCode": null,
    "requestId": 1075921,
    "sessionId": null,
    "errorMessage": ""
}
```

#### dataset-order-products

Lists all available order products for a dataset - this does not guarantee scene availability.

##### Input Parameters

| Parameter Name | Data Type | Required | Description |
| --- | --- | --- | --- |
| datasetName | string | Yes | Used to identify the which dataset to return results for |
| Parameter Name | Data Type | Required | Description |

##### Examples

**Sample Request**  
```
{
    "datasetName": "gls_all"
}
```

**Sample Response**  
```
{
    "data": [
        {
            "productCode": "W005",
            "productName": "GLS WMS ONDEMAND"
        }
    ],
    "version": "stable",
    "errorCode": null,
    "requestId": 28046,
    "sessionId": null,
    "errorMessage": ""
}
```

This method is used to find datasets available for searching. By passing only API Key, all available datasets are returned. Additional parameters such as temporal range and spatial bounding box can be used to find datasets that provide more specific data. The dataset name parameter can be used to limit the results based on matching the supplied value against the public dataset name with assumed wildcards at the beginning and end.

##### Input Parameters

| Parameter Name | Data Type | Required | Description |
| --- | --- | --- | --- |
| catalog | string | No | Used to identify datasets that are associated with a given application |
| categoryId | string | No | Used to restrict results to a specific category (does not search sub-sategories) |
| datasetName | string | No | Used as a filter with wildcards inserted at the beginning and the end of the supplied value |
| includeMessages | boolean | No | Optional parameter to include messages regarding specific dataset components |
| publicOnly | boolean | No | Used as a filter out datasets that are not accessible to unauthenticated general public users |
| includeUnknownSpatial | boolean | No | Optional parameter to include datasets that do not support geographic searching |
| temporalFilter | [TemporalFilter](https://m2m.cr.usgs.gov/api/docs/datatypes/#temporalFilter) | No | Used to filter data based on data acquisition |
| spatialFilter | [SpatialFilter](https://m2m.cr.usgs.gov/api/docs/datatypes/#spatialFilter) | No | Used to filter data based on data location |
| sortDirection | string | No | Defined the sorting as Ascending (ASC) or Descending (DESC) - default is ASC |
| sortField | string | No | Identifies which field should be used to sort datasets (shortName - default, longName, dastasetName, GloVis) |
| useCustomization | boolean | No | Optional parameter to indicate whether to use customization |
| Parameter Name | Data Type | Required | Description |

##### Examples

**Sample Request**  
```
{
    "datasetName": "Global Land Survey",
    "spatialFilter": {
        "filterType": "mbr",
        "lowerLeft": {
                "latitude": 44.60847,
                "longitude": -99.69639
        },
        "upperRight": {
                "latitude": 44.60847,
                "longitude": -99.69639
        }
    },
    "temporalFilter": {
        "start": "2012-01-01",
        "end": "2012-12-01"
    }
}
```

**Sample Response**  
```
{
    "data": [
        {
            "catalogs": [
                "EE",
                "GV"
            ],
            "keywords": "Landsat 7 Enhanced Thematic Mapper Plus (ETM+),Global Land Survey (GLS),Landsat,Satellite Imagery,Remotely Sensed Imagery,Landsat 5 Thematic Mapper (TM),Visible Wavelengths,Thermal Wavelengths,Infrared Wavelengths",
            "legacyId": 13869,
            "dataOwner": "DMID",
            "datasetId": "5e7c4182eba11e53",
            "doiNumber": "https:\/\/doi.org\/10.5066\/F7M32TQB",
            "sceneCount": 42992,
            "dateUpdated": "2020-03-26 00:45:38.587013-05",
            "abstractText": "The Global Land Survey (GLS) datasets are a collection of orthorectified, cloud-minimized Landsat-type satellite images, providing near complete coverage of the global land area decadally since the early 1970s.  The global mosaics are centered on 1975, 1990, 2000, 2005, and 2010, and consist of data acquired from five sensors: Operational Land Imager, Enhanced Thematic Mapper Plus, Thematic Mapper, Multispectral Scanner, and Advanced Land Imager.  This newest version combines all of the GLS data into one collection which has all of the combined collections.  The GLS datasets have been widely used in land-cover and land-use change studies at local, regional, and global scales.  This study evaluates the GLS datasets with respect to their spatial coverage, temporal consistency, geodetic accuracy, radiometric calibration consistency, image completeness, extent of cloud contamination, and residual gaps.  The datasets have been improved in order to give spatial continuity across all decadal collections.  Most of the imagery (85%) having cloud cover of less than 10%, the acquisition years clustered much more tightly around their target years, better co-registration relative to GLS-2000, and better radiometric absolute calibration.  Probably, the most significant impediment to scientific use of the datasets is the variability of image phenology (i.e., acquisition day of year).  This collection provides end-users with an assessment of the quality of the GLS datasets for specific applications, and where possible, suggestions for mitigating their deficiencies.",
            "datasetAlias": "gls_all",
            "spatialBounds": {
                "east": 180.01,
                "west": -180.01,
                "north": 82.6855268254304,
                "south": -69.3502337564158
            },
            "acquisitionEnd": "2012-01-30",
            "collectionName": "Global Land Survey",
            "ingestFrequency": null,
            "acquisitionStart": "1972-07-25",
            "temporalCoverage": "[\"1972-07-25 00:00:00-05\",\"2012-01-30 00:00:00-06\"]",
            "supportCloudCover": true,
            "collectionLongName": "Global Land Survey",
            "datasetCategoryName": "Landsat Legacy",
            "supportDeletionSearch": false
        }
    ],
    "version": "stable",
    "errorCode": null,
    "requestId": 123456,
    "sessionId": 123456,
    "errorMessage": ""
}
```

#### dataset-set-customization

This method is used to create or update dataset customizations for a given dataset.

##### Input Parameters

| Parameter Name | Data Type | Required | Description |
| --- | --- | --- | --- |
| datasetName | string | Yes | Used to identify the dataset to search |
| excluded | boolean | No | Used to exclude the dataset |
| metadata | [Metadata](https://m2m.cr.usgs.gov/api/docs/datatypes/#metadata) | No | Used to customize the metadata layout. |
| searchSort | [SearchSort](https://m2m.cr.usgs.gov/api/docs/datatypes/#searchSort) | No | Used to sort the dataset results. |
| fileGroups | [FileGroups](https://m2m.cr.usgs.gov/api/docs/datatypes/#fileGroups) | No | Used to customize downloads by file groups |
| Parameter Name | Data Type | Required | Description |

##### Examples

**Sample Request**  
```
{
    "excluded": false,
    "metadata": {
        "full": [
            {
                "id": "5e7c4182f19ae74c",
                "sortOrder": 1
            },
            {
                "id": "5e7c4182852f34b2",
                "sortOrder": 2
            },
            {...},
            {...}
        ],
        "res_sum": [
            {
                "id": "5e7c418236ae7c32",
                "sortOrder": 1
            },
            {...},
            {...}
        ]
    },
    "searchSort": [
        {
            "id": "5e7c418236ae7c32",
            "direction": "ASC"
        },
        {
            "id": "5e7c41824cb1f9ed",
            "direction": "DESC"
        },
        {...}
    ],
    "fileGroups": {"group_id": ["product_id"]}
    "datasetName": "gls_all"
}
```

**Sample Response**  
```
{
    "data": 1,
    "version": "stable",
    "errorCode": null,
    "requestId": 1077205,
    "sessionId": null,
    "errorMessage": ""
}
```

#### dataset-set-customizations

This method is used to create or update customizations for multiple datasets at once.

##### Input Parameters

| Parameter Name | Data Type | Required | Description |
| --- | --- | --- | --- |
| datasetCustomization | [DatasetCustomization](https://m2m.cr.usgs.gov/api/docs/datatypes/#datasetCustomization) | Yes | Used to create or update a dataset customization for multiple datasets. |
| Parameter Name | Data Type | Required | Description |

##### Examples

**Sample Request**  
```
{
    "datasetCustomization": {
        "gls_all": [
            {
                "excluded": false,
                "metadata": {
                    "full": [
                        {
                            "id": "5e7c4182f19ae74c",
                            "sortOrder": 1
                        },
                        {...},
                        {...}
                    ],
                    "export": [
                        {
                            "id": "5e7c418236ae7c32",
                            "sortOrder": 1
                        },
                        {...}
                    ]
                },
                "search_sort": [
                    {
                        "id": "5e7c418236ae7c32",
                        "direction": "DESC"
                    }
                ]
            }
        ],
        "asas": [
            {
                "excluded": true,
                "metadata": {
                    "full": [
                        {
                            "id": "16-char-id",
                            "sortOrder": 1
                        }
                    ],
                    "export": [
                        {
                            "id": "16-char-id",
                            "sortOrder": 1
                        }
                    ]
                },
                "search_sort": [
                    {
                        "id": "16-char-id",
                        "direction": "ASC"
                    }
                ]
            }
        ],
        "landsat_band_files_c2_l1": [
            {
                "excluded": False,
                "fileGroups": {
                    "ls_c2l1_all": [
                        "63ceb05468b7e8b3"
                    ], 
                    "ls_c2l1_ot_band": [
                        "63ceb05468b7e8b4"
                    ]
                }
            }
        ],
    }
}
```

**Sample Response**  
```
{
    "data": 1,
    "version": "stable",
    "errorCode": null,
    "requestId": 1077476,
    "sessionId": null,
    "errorMessage": ""
}
```

#### download-complete-proxied

Updates status to 'C' with total downloaded file size for completed proxied downloads

##### Input Parameters

| Parameter Name | Data Type | Required | Description |
| --- | --- | --- | --- |
| proxiedDownloads | [ProxiedDownload\[\]](https://m2m.cr.usgs.gov/api/docs/datatypes/#proxiedDownload[]) | Yes | Used to specify multiple proxied downloads |
| Parameter Name | Data Type | Required | Description |

##### Examples

**Sample Request**  
```
{
    "proxiedDownloads": [
        {
            "downloadId": 3046459,
            "downloadedSize": 55556
        },
        {
            "downloadId": 3046460,
            "downloadedSize": 55557
        },
        {
            "downloadId": 3046458,
            "downloadedSize": 55558
        }
    ]
}
```

**Sample Response**  
```
{
    "data": {
        "failed": [],
        "updatedDownloadsCount": 3
    },
    "version": "stable",
    "errorCode": null,
    "requestId": 123456,
    "sessionId": 123456,
    "errorMessage": ""
}
```

#### download-eula

Gets the contents of a EULA from the eulaCodes.

##### Input Parameters

| Parameter Name | Data Type | Required | Description |
| --- | --- | --- | --- |
| eulaCode | string | No | Used to specify a single eula |
| eulaCodes | string\[\] | No | Used to specify multiple eulas |
| Parameter Name | Data Type | Required | Description |

##### Examples

**Sample Request**  
```
{
    "eulaCodes": [
        "HDDS"
    ]
}
```

**Sample Response**  
```
{
    "data": [
        {
            "eulaCode": "HDDS",
            "agreementContent": "You have selected a data set that may include usage restrictions.  \nThese data are subject to the terms and conditions specified in \nthe accompanying license.\n\nUse of the data in published work shall include the copyright\/logo \nfrom the original data provider.  Any data redistributed to licensed \nusers shall include a copy of the license agreement. \n\nUSGS will not be held responsible or liable for misuse or \nmisrepresentation by the end users and is not required to enforce \nthese provisions beyond communicating them to the user.  It is the \nresponsibility of the user to adhere to these terms and conditions. \n\nYou must agree to these terms to proceed.\n"
        }
    ],
    "version": "stable",
    "errorCode": null,
    "requestId": 20086,
    "sessionId": 1690,
    "errorMessage": ""
}
```

Gets a list of unique download labels associated with the orders.

##### Input Parameters

| Parameter Name | Data Type | Required | Description |
| --- | --- | --- | --- |
| downloadApplication | string | No | Used to denote the application that will perform the download |
| Parameter Name | Data Type | Required | Description |

##### Examples

**Sample Request**  
```
{
    "downloadApplication": "BulkDownload"
}
```

**Sample Response**  
```
{
    "data": [
        {
            "label": "123456",
            "dateEntered": 1583859782139,
            "downloadSize": null,
            "downloadCount": 0,
            "totalComplete": 8
        },
        {
            "label": "1234567",
            "dateEntered": 1584936362122,
            "downloadSize": null,
            "downloadCount": 0,
            "totalComplete": 7
        }        
    ],
    "version": "stable",
    "errorCode": null,
    "requestId": 28081,
    "sessionId": null,
    "errorMessage": ""
}
```

#### download-options

The download options request is used to discover downloadable products for each dataset. If a download is marked as not available, an order must be placed to generate that product.

##### Input Parameters

| Parameter Name | Data Type | Required | Description |
| --- | --- | --- | --- |
| datasetName | string | Yes | Dataset alias |
| entityIds | string | No | List of scenes |
| listId | string | No | Used to identify the list of scenes to use |
| includeSecondaryFileGroups | boolean | No | Optional parameter to return file group IDs with secondary products |
| Parameter Name | Data Type | Required | Description |

##### Examples

**Sample Request**  
```
{
    "datasetName": "landsat_ot_c2_l1",
    "entityIds": "LC90330412023209LGN09",
    "includeSecondaryFileGroups": true
}
Note: "listId" is the id of the customized list which is built by scene-list-add. 
The parameter entityIds can be either a string array or a string. If passing them in a string, separate them by comma (no space between the IDs).
If passing them in the test page, use string without quotes/spaces/brackets, just pass entityIds with commas, for example, 
LT50290302005219EDC00,LE70820552011359EDC00
```

**Sample Response**  
```
{
    "requestId": 17774310,
    "version": "stable",
    "data": [
        {
            "id": "73ceb05468b7e8c2",
            "downloadName": "C2L1 Tile Product Files",
            "displayId": "LC09_L1TP_033041_20230728_20230830_02_T1",
            "entityId": "LC90330412023209LGN09",
            "datasetId": "5e81f14f59432a27",
            "available": true,
            "filesize": 0,
            "productName": "Landsat Collection 2 Level-1 Band File",
            "productCode": "D687",
            "bulkAvailable": true,
            "downloadSystem": "folder",
            "secondaryDownloads": [
                {
                    "id": "gy7bvmukhtkl910o",
                    "downloadName": "ANG.txt",
                    "displayId": "LC09_L1TP_033041_20230728_20230830_02_T1_ANG.txt",
                    "entityId": "L1_LC09_L1TP_033041_20230728_20230830_02_T1_ANG_TXT",
                    "datasetId": "5e81ff5f86d11d08",
                    "available": true,
                    "filesize": 117391,
                    "productName": "Landsat Collection 2 Level-1 Band File",
                    "productCode": "D687",
                    "bulkAvailable": true,
                    "downloadSystem": "dds",
                    "secondaryDownloads": [],
                    "fileGroups": [
                        "ls_c2l1_all"
                    ],
                    "displayOrder": 6
                },
                ...
            ]
        },
        ...
    ],
    "errorCode": null,
    "errorMessage": null
}
If available checksum will be included in the response which can be used to validate the downloaded file.
Sample Format:
"checksum": [
                {
                    "id": "cksum",
                    "value": 1876874104
                },
                {
                    "id": "sha256sum",
                    "value": "c5c6efd960a735ac49de4717668a7472681b6c8cf2d53292a6de822f911ebe63"
                }
            ]
```

#### download-order-load

This method is used to prepare a download order for processing by moving the scenes into the queue for processing

##### Input Parameters

| Parameter Name | Data Type | Required | Description |
| --- | --- | --- | --- |
| downloadApplication | string | No | Used to denote the application that will perform the download |
| label | string | No | Determines which order to load |
| Parameter Name | Data Type | Required | Description |

##### Examples

**Sample Request**  
```
{
    "label": [
        "123456"
    ],
    "downloadApplication": "BulkDownload"
}
```

**Sample Response**  
```
{
    "data": [
        {
            "label": "123456",
            "entityId": "NB1NHAP840240229",
            "eulaCode": null,
            "filesize": 13755824,
            "datasetId": "5e83a328548fe769",
            "displayId": "NB1NHAP840240229",
            "downloadId": 1625144,
            "statusCode": "C",
            "statusText": "Complete",
            "productCode": "D132",
            "productName": "NATIONAL HIGH ALTITUDE PROGRAM\/ALASKA HIGH ALTITUDE PHOTOGRAPHY(NHAP\/AHAP) MEDIUM RESOLUTION DOWNLOAD",
            "collectionName": "NHAP"
        }  
    ],
    "version": "stable",
    "errorCode": null,
    "requestId": 25582,
    "sessionId": 1690,
    "errorMessage": ""
}
```

#### download-order-remove

This method is used to remove an order from the download queue.

##### Input Parameters

| Parameter Name | Data Type | Required | Description |
| --- | --- | --- | --- |
| downloadApplication | string | No | Used to denote the application that will perform the download |
| label | string | Yes | Determines which order to remove |
| Parameter Name | Data Type | Required | Description |

##### Examples

**Sample Request**  
```
{
    "label": "123456",
    "downloadApplication": "BulkDownload"
}
```

**Sample Response**  
```
{
    "data": 2,
    "version": "stable",
    "errorCode": null,
    "requestId": 20017,
    "sessionId": 1743,
    "errorMessage": ""
}
```

#### download-remove

Removes an item from the download queue.

##### Input Parameters

| Parameter Name | Data Type | Required | Description |
| --- | --- | --- | --- |
| downloadId | int | Yes | Represents the ID of the download from within the queue |
| Parameter Name | Data Type | Required | Description |

##### Examples

**Sample Request**  
```
{
    "downloadId": "1632783"
}
Note: "downloadId" can be retrieved by calling download-search
```

**Sample Response**  
```
{
    "data": true,
    "version": "stable",
    "errorCode": null,
    "requestId": 26094,
    "sessionId": 1690,
    "errorMessage": ""
}
```

#### download-request

This method is used to insert the requested downloads into the download queue and returns the available download URLs.  
  
Each ID supplied in the downloads parameter you provide will be returned in one of three elements:
- availableDownloads - URLs provided in this list are immediately available; note that these URLs take you to other distribution systems that may require authentication
- preparingDownloads - IDs have been accepted but the URLs are NOT YET available for use
- failed - IDs were rejected; see the errorMessage field for an explanation

Other information is also provided in the response:

- newRecords - Includes a downloadId for each element of the downloads parameter that was accepted and a label that applies to the whole request
- duplicateProducts - Requests that duplicate previous requests by the same user; these are not re-added to the queue and are not included in newRecords
- numInvalidScenes - The number of products that could not be found by ID or failed to be requested for any reason
- remainingLimits - The number of remaining downloads to hit the rate limits by user and IP address
- limitType - The type of the limits are counted by, the value is either 'user' or 'ip'
	- username - The user name associated with the request
	- ipAddress - The IP address associated with the request
	- recentDownloadCount - The number of downloads requested in the past 15 minutes
	- pendingDownloadCount - The number of downloads in pending state before they are available for download
	- unattemptedDownloadCount - The number of downloads in available status but the user has not downloaded yet

##### Input Parameters

| Parameter Name | Data Type | Required | Description |
| --- | --- | --- | --- |
| configurationCode | string | No | Used to customize the the download routine, primarily for testing. The valid values include no\_data, test, order, order+email and null |
| downloadApplication | string | No | Used to denote the application that will perform the download (default = M2M). Internal use only. |
| downloads | [Download\[\]](https://m2m.cr.usgs.gov/api/docs/datatypes/#download[]) | No | Used to identify higher level products that this data may be used to create |
| dataPaths | [FilepathDownload\[\]](https://m2m.cr.usgs.gov/api/docs/datatypes/#filepathDownload[]) | No | Used to identify products by data path, specifically for internal automation and DDS functionality |
| label | string | No | If this value is passed it will overide all individual download label values |
| systemId | string | No | Identifies the system submitting the download/order (default = M2M). Internal use only |
| dataGroups | [FilegroupDownload\[\]](https://m2m.cr.usgs.gov/api/docs/datatypes/#filegroupDownload[]) | No | Identifies the products by file groups |
| Parameter Name | Data Type | Required | Description |

##### Examples

**Sample Request**  
```
* Download by downloads
{
    "downloads": [
        {
            "label": "123456",
            "entityId": "LC80030712017086LGN00",
            "productId": "5e9eb2249228fe8f"
        }
    ],
    "downloadApplication": "EE"
}

 * Download by file groups
{
    "dataGroups": [
        {
            "label": "my_label",
            "datasetName": "landsat_band_files_c2_l1",
            "fileGroups": ["ls_c2l1_ot_band"]
        }
    ],
    "downloadApplication": "M2M"
}
```

**Sample Response**  
```
{
    "data": {
        "failed": [],
        "newRecords": {
            "1738054": "20263459"
        },
        "remainingLimits": [
            {
                "username": "user_name",
                "limitType": "user",
                "recentDownloadCount": 14999,
                "pendingDownloadCount": 19999,
                "unattemptedDownloadCount": 19997
            },
            {
                "ipAddress": "110.209.64.55",
                "limitType": "ip",
                "recentDownloadCount": 14999,
                "pendingDownloadCount": 19999,
                "unattemptedDownloadCount": 20000
            }
        ],
        "numInvalidScenes": 0,
        "duplicateProducts": [],
        "availableDownloads": [],
        "preparingDownloads": [
            {
                "url": "https:\/\/dds.cr.usgs.gov\/download-staging\/eyJpZCI6MjAyNjM0NTksImNvbnRhY3RJZCI6NDE5NzY0fQ==",
                "eulaCode": null,
                "entityId": "LC80030712017086LGN00",
                "downloadId": 1738054
            }
        ]
    },
    "version": "development",
    "errorCode": null,
    "requestId": 123456789,
    "sessionId": 2456789,
    "errorMessage": ""
}
If available checksum_values will be included in the response which can be used to validate the downloaded file.
Sample Format:
"checksum_values": [
                    {
                        "id": "cksum",
                        "value": 3119187019
                    },
                    {
                        "id": "sha256sum",
                        "value": "36197d26f360d5b24f702aab0998c2d4a50de5ec75f1a8a506123d882874a47f"
                    }
                ]
```

#### download-retrieve

Returns all available and previously requests but not completed downloads.  
  

##### Input Parameters

| Parameter Name | Data Type | Required | Description |
| --- | --- | --- | --- |
| downloadApplication | string | No | Used to denote the application that will perform the download |
| label | string | No | Determines which downloads to return |
| Parameter Name | Data Type | Required | Description |

##### Examples

**Sample Request**  
```
{
    "label": "test",
    "downloadApplication": "BulkDownload"
}
```

**Sample Response**  
```
{
    "data": {
        "eulas": [],
        "available": [
            {
                "url": "https:\/\/dds.cr.usgs.gov\/download\/eyJpZCI6MTYyMDk3MiwiY29udGFjdElkIjozMTQ0NTh9",
                "label": "test",
                "entityId": "26162",
                "eulaCode": null,
                "filesize": 622623982,
                "datasetId": "5e83a42ca9977c30",
                "displayId": "S2A_OPER_MSI_L1C_TL_MTI__20160116T172531_20160116T203906_A002969_T15TUE_N02_01_01",
                "downloadId": 1620972,
                "statusCode": "A",
                "statusText": "Available",
                "productCode": "D557",
                "productName": "SENTINEL 2 DOWNLOAD",
                "collectionName": "Sentinel-2"
            }
        ],
        "queueSize": 1,
        "requested": []
    },
    "version": "stable",
    "errorCode": null,
    "requestId": 12345,
    "sessionId": 1906,
    "errorMessage": ""
}
If available checksum values will be included in the response which can be used to validate the downloaded file.
Sample Format:
"checksum": [
                {
                    "id": "cksum",
                    "value": 803788146
                }
            ],
```

This method is used to search for downloads within the queue, regardless of status, that match the given label.

##### Input Parameters

| Parameter Name | Data Type | Required | Description |
| --- | --- | --- | --- |
| activeOnly | boolean | No | Determines if completed, failed, cleared and proxied downloads are returned |
| label | string | No | Used to filter downloads by label |
| downloadApplication | string | No | Used to filter downloads by the intended downloading application |
| Parameter Name | Data Type | Required | Description |

##### Examples

**Sample Request**  
```
{
    "label": "test",
    "activeOnly": false,
    "downloadApplication": "BulkDownload"
}
```

**Sample Response**  
```
{
    "data": [
        {
            "label": "test",
            "entityId": "EO1A0010162001148111PP_SGS_01",
            "eulaCode": null,
            "filesize": 569406,
            "datasetId": "5e839cb7c77529ab",
            "displayId": "EO1A0010162001148111PP_SGS_01",
            "downloadId": 1620031,
            "statusCode": "P",
            "statusText": "Proxied",
            "productCode": "D408",
            "productName": "EO-1 ALI GIS READY BUNDLE DOWNLOAD",
            "collectionName": "EO-1 ALI"
        }        
    ],
    "version": "stable",
    "errorCode": null,
    "requestId": 16962,
    "sessionId": 1664,
    "errorMessage": ""
}
```

#### download-summary

Gets a summary of all downloads, by dataset, for any matching labels.

##### Input Parameters

| Parameter Name | Data Type | Required | Description |
| --- | --- | --- | --- |
| downloadApplication | string | Yes | Used to denote the application that will perform the download |
| label | string | Yes | Determines which downloads to return |
| sendEmail | boolean | No | If set to true, a summary email will also be sent |
| Parameter Name | Data Type | Required | Description |

##### Examples

**Sample Request**  
```
{
    "label": "test",
    "sendEmail": false,
    "downloadApplication": "BulkDownload"
}
```

**Sample Response**  
```
{
    "data": {
        "label": "test",
        "sceneCount": 13,
        "collections": [
            {
                "sceneCount": 13,
                "datasetName": "declassiii",
                "downloadCount": 13,
                "collectionName": "Declass 3 (2013)",
                "totalEstimatedSize": "42276114112"
            }
        ],
        "downloadCount": 13,
        "totalEstimatedSize": 42276114112
    },
    "version": "stable",
    "errorCode": null,
    "requestId": 5471,
    "sessionId": null,
    "errorMessage": ""
}
```

#### grid2ll

Used to translate between known grids and coordinates.

##### Input Parameters

| Parameter Name | Data Type | Required | Description |
| --- | --- | --- | --- |
| gridType | string | Yes | Which grid system is being used? (WRS1 or WRS2) |
| responseShape | string | No | What type of geometry should be returned - a bounding box polygon or a center point? (polygon or point) |
| path | string | No | The x coordinate in the grid system |
| row | string | No | The y coordinate in the grid system |
| Parameter Name | Data Type | Required | Description |

##### Examples

**Sample Response**  
```
{
    "data": {
        "shape": "point",
        "coordinates": [
            {
                "latitude": 76.47055372032352,
                "longitude": -24.019378579142632
            }
        ]
    },
    "version": "stable",
    "errorCode": null,
    "requestId": 38333,
    "sessionId": null,
    "errorMessage": ""
}
```

#### login-app-guest

This endpoint assumes that the calling application has generated a single-use token to complete the authentication and return an API Key specific to that guest user. All subsequent requests should use the API Key under the 'X-Auth-Token' HTTP header as the Single Sign-On cookie will not authenticate those requests. The API Key will be active for two hours, which is restarted after each subsequent request, and should be destroyed upon final use of the service by calling the logout method.  
  
The 'appToken' field will be used to verify the 'Referrer' HTTP Header to ensure the request was authentically sent from the assumed application.

##### Input Parameters

| Parameter Name | Data Type | Required | Description |
| --- | --- | --- | --- |
| applicationToken | string | Yes | The token for the calling application |
| userToken | string | Yes | The single-use token generated for this user |
| Parameter Name | Data Type | Required | Description |

##### Examples

**Sample Request**  
```
{"applicationToken": "foo", "userToken": "bar"}
```

**Sample Response**  
```
{"errorCode": null, "errorMessage": "", "data": "eyJjaWQiOjQxOTY3NCwicyI6IjE1NzkwNjM0NzAiLCJwZXJtaXNzaW9ucyI6W119", "requestId": 123, "sessionId": 18}
```

#### login-sso

This endpoint assumes that a user has an active ERS Single Sign-On Cookie in their browser or attached to this request. Authentication will be performed from the Single Sign-On Cookie and return an API Key upon successful authentication. All subsequent requests should use the API Key under the 'X-Auth-Token' HTTP header as the Single Sign-On cookie will not authenticate those requests. The API Key will be active for two hours, which is restarted after each subsequent request, and should be destroyed upon final use of the service by calling the logout method.

##### Input Parameters

| Parameter Name | Data Type | Required | Description |
| --- | --- | --- | --- |
| userContext | [UserContext](https://m2m.cr.usgs.gov/api/docs/datatypes/#userContext) | No | Metadata describing the user the request is on behalf of |
| Parameter Name | Data Type | Required | Description |

##### Examples

**Sample Request**  
```
{}
```

**Sample Response**  
```
{
    "data": {
        "apiKey": "eyJjaWQiOjQxOTY3NCwicyI6IjE1NzkwNjM0NzAiLCJwZXJtaXNzaW9ucyI6W119",
        "username": "foo"
    },
    "version": "stable",
    "errorCode": null,
    "requestId": 1234,
    "sessionId": 1085,
    "errorMessage": ""
}
```

#### login-token

This login method uses ERS application tokens to allow for authentication that is not directly tied the users ERS password. Instructions for generating the application token can be found [here](https://www.usgs.gov/media/files/m2m-application-token-documentation).  
  
Upon a successful login, an API key will be returned. This key will be active for two hours and should be destroyed upon final use of the service by calling the logout method.  
  
**This request requires an HTTP POST request instead of a HTTP GET request as a security measure to prevent username and password information from being logged by firewalls, web servers, etc.**

##### Input Parameters

| Parameter Name | Data Type | Required | Description |
| --- | --- | --- | --- |
| username | string | Yes | ERS Username |
| token | string | Yes | Application Token |
| userContext | [UserContext](https://m2m.cr.usgs.gov/api/docs/datatypes/#userContext) | No | Metadata describing the user the request is on behalf of |
| Parameter Name | Data Type | Required | Description |

##### Examples

**Sample Request**  
```
{"username": "foo", "token": "bar"}
```

**Sample Response**  
```
{"errorCode": null, "errorMessage": "", "data": "eyJjaWQiOjQxOTY3NCwicyI6IjE1NzkwNjM0NzAiLCJwZXJtaXNzaW9ucyI6W119", "requestId": 123, "sessionId": 18}
```

#### logout

This method is used to remove the users API key from being used in the future.

##### Input Parameters

<table><caption>Displays logout Input Parameters</caption><thead><tr><th>Parameter Name</th><th>Data Type</th><th>Required</th><th>Description</th></tr></thead><tbody><tr><td colspan="4">No Parameters for Endpoint</td></tr></tbody><tfoot><tr><td>Parameter Name</td><td>Data Type</td><td>Required</td><td>Description</td></tr></tfoot></table>

##### Examples

This request does not use request parameters and does not return a data value.

#### notifications

Gets a notification list.

##### Input Parameters

| Parameter Name | Data Type | Required | Description |
| --- | --- | --- | --- |
| systemId | string | Yes | Used to identify notifications that are associated with a given application |
| Parameter Name | Data Type | Required | Description |

##### Examples

**Sample Request**  
```
{
    "systemId": "EE"
}
```
Note: Few valid systems ids are BDA, DDS, EE, ERS, GVN, HDDS, M2M, etc.

**Sample Response**  
```
{
    "data": [
        {
            "id": 46,
            "subject": "Test Informational Notification",
            "dateUpdated": "2020-09-18 14:57:00-05",
            "severityCode": "I",
            "severityText": "Informational",
            "messageContent": "This is a INFORMATIONAL test notification.",
            "severityCssClass": "alert-info"
        }
    ],
    "version": "stable",
    "errorCode": null,
    "requestId": 27726,
    "sessionId": 1932,
    "errorMessage": ""
}
```

#### order-products

Gets a list of currently selected products - paginated.

##### Input Parameters

| Parameter Name | Data Type | Required | Description |
| --- | --- | --- | --- |
| datasetName | string | Yes | Dataset alias |
| entityIds | string | No | List of scenes |
| listId | string | No | Used to identify the list of scenes to use |
| Parameter Name | Data Type | Required | Description |

##### Examples

**Sample Request**  
```
{
    "listId": "my_order_list",
    "datasetName": "ard_tile"
}
Note: "listId" is the id of the customized list which is built by scene-list-add
```

**Sample Response**  
```
{
    "data": [
        {
            "id": "5e83a38b8a9578c2",
            "price": "0",
            "entityId": "LE07_CU_016004_20180206_C01_V01",
            "available": true,
            "datasetId": "5e83a38b677b457d",
            "productCode": "W018",
            "productName": "ARD TILE WMS ONDEMAND"
        },
        {
            "id": "5e83a38b8a9578c2",
            "price": "0",
            "entityId": "LE07_AK_008009_20161012_C01_V01",
            "available": true,
            "datasetId": "5e83a38b677b457d",
            "productCode": "W018",
            "productName": "ARD TILE WMS ONDEMAND"
        }
    ],
    "version": "stable",
    "errorCode": null,
    "requestId": 26293,
    "sessionId": 1878,
    "errorMessage": ""
}
```

#### order-submit

Submits the current product list as a TRAM order - internally calling tram-order-create.

##### Input Parameters

| Parameter Name | Data Type | Required | Description |
| --- | --- | --- | --- |
| autoBulkOrder | boolean | No | If any products can be bulk ordered as a resulk of completed processing this option allows users to have orders automatically submitted. |
| products | [Product\[\]](https://m2m.cr.usgs.gov/api/docs/datatypes/#product[]) | Yes | Used to identify higher level products that this data may be used to create |
| processingParameters | string | No | Optional processing parameters to send to the processing system |
| priority | int | No | Processing Priority |
| orderComment | string | No | Optional textual identifier for the order |
| systemId | string | No | Identifies the system submitting the order |
| Parameter Name | Data Type | Required | Description |

##### Examples

**Sample Request**  
```
{
    "products": [
        {
            "entityId": "D3C1218-401766A004",
            "productId": "5e7c41f37c601343"
        },
        {
            "entityId": "D3C1218-401766A003",
            "productId": "5e7c41f37c601343"
        }
    ],
    "orderComment": null,
    "autoBulkOrder": false
}
```

**Sample Response**  
```
{
    "data": {
        "failed": [],
        "orderNumbers": [
            "123456"
        ]
    },
    "version": "stable",
    "errorCode": null,
    "requestId": 15193,
    "sessionId": 1589,
    "errorMessage": ""
}
```

#### permissions

Returns a list of user permissions for the authenticated user. This method does not accept any input.

##### Input Parameters

<table><caption>Displays permissions Input Parameters</caption><thead><tr><th>Parameter Name</th><th>Data Type</th><th>Required</th><th>Description</th></tr></thead><tbody><tr><td colspan="4">No Parameters for Endpoint</td></tr></tbody><tfoot><tr><td>Parameter Name</td><td>Data Type</td><td>Required</td><td>Description</td></tr></tfoot></table>

##### Examples

**Sample Response**  
```
{"errorCode": null, "errorMessage": "", "data": ["download", "order"], "requestId": 123, "sessionId": 18}
```

#### rate-limit-summary

Returns download rate limits and how many downloads are in each status as well as how close the user is to reaching the rate limits  
  

Three elements are provided in the response:

- initialLimits - Includes the initial downloads rate limits
- recentDownloadCount - The maximum number of downloads requested in the past 15 minutes
	- pendingDownloadCount - The maximum number of downloads in pending state before they are available for download
	- unattemptedDownloadCount - The maximum number of downloads in available status but the user has not downloaded yet
- remainingLimits - - Includes downloads that are currently remaining and count towards the rate limits. Users should be watching out for any of those numbers approaching 0 which means it is close to hitting the rate limits
- limitType - The type of the limits are counted by, the value is either 'user' or 'ip'
	- username - The user name associated with the request
	- ipAddress - The IP address associated with the request
	- recentDownloadCount - The number of downloads requested in the past 15 minutes
	- pendingDownloadCount - The number of downloads in pending state before they are available for download
	- unattemptedDownloadCount - The number of downloads in available status but the user has not downloaded yet
- recentDownloadCounts - Includes the downloads count in each status for the past 15 minutes
- countType - The type of the download counts are calculated by, the value is either 'user' or 'ip'
	- username - The user name associated with the request
	- ipAddress - The IP address associated with the request
	- downloadCount - The number of downloads per status in the past 15 minutes

##### Input Parameters

| Parameter Name | Data Type | Required | Description |
| --- | --- | --- | --- |
| ipAddress | string\[\] | No | Used to specify multiple IP address |
| Parameter Name | Data Type | Required | Description |

##### Examples

**Sample Request**  
```
{}
```

**Sample Response**  
```
{
    "requestId": 18965285,
    "version": "stable",
    "data": {
        "initialLimits": [
            {
                "recentDownloadCount": 15000,
                "pendingDownloadCount": 20000,
                "unattemptedDownloadCount": 20000
            }
        ],
        "remainingLimits": [
            {
                "limitType": "user",
                "username": "mwang",
                "recentDownloadCount": 14999,
                "pendingDownloadCount": 20000,
                "unattemptedDownloadCount": 19995
            },
            {
                "limitType": "ip",
                "ipAddress": "10.209.64.55",
                "recentDownloadCount": 14999,
                "pendingDownloadCount": 20000,
                "unattemptedDownloadCount": 19998
            }
        ],
        "recentDownloadCounts": [
            {
                "countType": "username",
                "username": "user_name",
                "downloadCounts": [
                    {
                        "status": "Available",
                        "downloadCount": 1
                    }
                ]
            },
            {
                "countType": "ip",
                "ipAddress": "10.209.64.55",
                "downloadCounts": [
                    {
                        "status": "Available",
                        "downloadCount": 1
                    }
                ]
            }
        ]
    },
    "errorCode": null,
    "errorMessage": null
}
```

#### scene-list-add

Adds items in the given scene list.

##### Input Parameters

| Parameter Name | Data Type | Required | Description |
| --- | --- | --- | --- |
| listId | string | Yes | User defined name for the list |
| datasetName | string | Yes | Dataset alias |
| idField | string | No | Used to determine which ID is being used - entityId (default) or displayId |
| entityId | string | No | Scene Identifier |
| entityIds | string\[\] | No | A list of Scene Identifiers |
| timeToLive | string | No | User defined lifetime using ISO-8601 formatted duration (such as "P1M") for the list |
| checkDownloadRestriction | boolean | No | Optional parameter to check download restricted access and availability |
| Parameter Name | Data Type | Required | Description |

##### Examples

**Sample Request**  
```
{
    "listId": "my_scene_list",
    "datasetName": "landsat_ot_c2_l2",
    "idField": "displayId",
    "entityId": "LC08_L2SP_012025_20201231_20210308_02_T1"
}
```

**Sample Response**  
```
Stable Branch:
{
    "requestId": 725466172,
    "version": "stable",
    "data": 1,
    "errorCode": null,
    "errorMessage": null
}

Development Branch:
{
    "data": {
        "failed": [
            {
                "entityId": "1044605",
                "licenseParameters": {
                    "event": "201605_Fire_Away"
                },
                "licensingRequired": "hdds_event"
            }
        ],
        "sceneCount": 0
    },
    "version": "development",
    "errorCode": null,
    "requestId": 1234567,
    "sessionId": 422421,
    "errorMessage": ""
}
```

#### scene-list-get

Returns items in the given scene list.

##### Input Parameters

| Parameter Name | Data Type | Required | Description |
| --- | --- | --- | --- |
| listId | string | Yes | User defined name for the list |
| datasetName | string | No | Dataset alias |
| startingNumber | int | No | Used to identify the start number to search from |
| maxResults | int | No | How many results should be returned? |
| Parameter Name | Data Type | Required | Description |

##### Examples

**Sample Request**  
```
{
    "listId": "my_scene_list"
}
```

**Sample Response**  
```
{
    "requestId": 725472107,
    "version": "stable",
    "data": [
        {
            "entityId": "LC80120252020366LGN00",
            "datasetName": "landsat_ot_c2_l2"
        }
    ],
    "errorCode": null,
    "errorMessage": null
}
```

#### scene-list-remove

Removes items from the given list. If no datasetName is provided, the call removes the whole list. If a datasetName is provided but no entityId, this call removes that dataset with all its IDs. If a datasetName and entityId(s) are provided, the call removes the ID(s) from the dataset.

##### Input Parameters

| Parameter Name | Data Type | Required | Description |
| --- | --- | --- | --- |
| listId | string | Yes | User defined name for the list |
| datasetName | string | No | Dataset alias |
| entityId | string | No | Scene Identifier |
| entityIds | string\[\] | No | A list of Scene Identifiers |
| Parameter Name | Data Type | Required | Description |

##### Examples

**Sample Request**  
```
{
    "listId": "my_scene_list",
    "datasetName": "landsat_ot_c2_l2",
    "entityId": "LC80120252020366LGN00"
}
```

**Sample Response**  
```
{
    "requestId": 725474929,
    "version": "stable",
    "data": null,
    "errorCode": null,
    "errorMessage": null
}
```

#### scene-list-summary

Returns summary information for a given list.

##### Input Parameters

| Parameter Name | Data Type | Required | Description |
| --- | --- | --- | --- |
| listId | string | Yes | User defined name for the list |
| datasetName | string | No | Dataset alias |
| Parameter Name | Data Type | Required | Description |

##### Examples

**Sample Request**  
```
{
    "listId": "my_scene_list"
}
```

**Sample Response**  
```
{
    "data": {
        "summary": {
            "sceneCount": "7",
            "spatialBounds": {
                "type": "Polygon",
                "coordinates": [
                    [
                        [
                            -98.7053,
                            42.11176
                        ],
                        [
                            -98.7053,
                            44.2421
                        ],
                        [
                            -95.65984,
                            44.2421
                        ],
                        [
                            -95.65984,
                            42.11176
                        ],
                        [
                            -98.7053,
                            42.11176
                        ]
                    ]
                ]
            },
            "temporalExtent": {
                "max": "2017-10-11 00:00:00-05",
                "min": "1999-12-05 00:00:00-06"
            }
        },
        "datasets": [
            {
                "sceneCount": 4,
                "datasetName": "lsr_landsat_etm_c1",
                "listTimeout": 1646991,
                "spatialBounds": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [
                                -98.7053,
                                42.2031
                            ],
                            [
                                -98.7053,
                                44.1573
                            ],
                            [
                                -95.65984,
                                44.1573
                            ],
                            [
                                -95.65984,
                                42.2031
                            ],
                            [
                                -98.7053,
                                42.2031
                            ]
                        ]
                    ]
                },
                "temporalExtent": {
                    "max": "2014-06-21 00:00:00-05",
                    "min": "1999-12-05 00:00:00-06"
                }
            }
        ]
    },
    "version": "stable",
    "errorCode": null,
    "requestId": 29235,
    "sessionId": 1981,
    "errorMessage": ""
}
```

#### scene-list-types

Returns scene list types (exclude, search, order, bulk, etc).

##### Input Parameters

| Parameter Name | Data Type | Required | Description |
| --- | --- | --- | --- |
| listFilter | string | No | If provided, only returns listIds that have the provided filter value within the ID |
| Parameter Name | Data Type | Required | Description |

##### Examples

**Sample Request**  
```
{}
```

**Sample Response**  
```
{
    "data": [
        {
            "list": " my_scene_list",
            "sceneCount": 3,
            "listTimeout": null
        }
    ],
    "version": "stable",
    "errorCode": null,
    "requestId": 28062,
    "sessionId": 1948,
    "errorMessage": ""
}
```

This request is used to return metadata for a given scene.

##### Input Parameters

| Parameter Name | Data Type | Required | Description |
| --- | --- | --- | --- |
| datasetName | string | Yes | Used to identify the dataset to search |
| entityId | string | Yes | Used to identify the scene to return results for |
| idType | string | No | If populated, identifies which ID field (entityId, displayId or orderingId) to use when searching for the provided entityId (default = entityId) |
| metadataType | string | No | If populated, identifies which metadata to return (summary, full, fgdc, iso) |
| includeNullMetadataValues | boolean | No | Optional parameter to include null metadata values |
| useCustomization | boolean | No | Optional parameter to display metadata view as per user customization |
| Parameter Name | Data Type | Required | Description |

##### Examples

**Sample Request**  
```
{
    "datasetName": "landsat_ot_c2_l2",
    "entityId": "LC08_L2SP_012025_20201231_20210308_02_T1",
    "idType": "displayId",
    "metadataType": "full",
    "useCustomization": false
}
```

**Sample Response**  
```
{
    "requestId": 725448618,
    "version": "stable",
    "data": {
        "browse": [
            {
                "id": "5fb4ba12d7ec307f",
                "browseRotationEnabled": false,
                "browseName": "Level-1 Reflective Browse",
                "browsePath": "https://landsatlook.usgs.gov/gen-browse?size=rrb&type=refl&product_id=LC08_L1TP_012025_20201231_20210308_02_T1",
                "overlayPath": "https://landsatlook.usgs.gov/dynamic-tiler/scenes/LC08_L1TP_012025_20201231_20210308_02_T1/tiles/{z}/{x}/{y}.png?layer=natural_color",
                "overlayType": "ls_chs",
                "thumbnailPath": "https://landsatlook.usgs.gov/gen-browse?size=thumb&type=refl&product_id=LC08_L1TP_012025_20201231_20210308_02_T1"
            }
        ],
        "cloudCover": "41.90",
        "entityId": "LC80120252020366LGN00",
        "displayId": "LC08_L2SP_012025_20201231_20210308_02_T1",
        "orderingId": "LC80120252020366LGN00",
        "metadata": [
            {
                "id": "5e83d1507bc900d5",
                "fieldName": "Landsat Product Identifier L2",
                "dictionaryLink": "https://www.usgs.gov/centers/eros/science/landsat-collection-2-data-dictionary\n#landsat_product_id",
                "value": "LC08_L2SP_012025_20201231_20210308_02_T1"
            },
            {
                "id": "5e83d15012b8905f",
                "fieldName": "Landsat Product Identifier L1",
                "dictionaryLink": "https://www.usgs.gov/centers/eros/science/landsat-collection-2-data-dictionary\n#landsat_product_id",
                "value": "LC08_L1TP_012025_20201231_20210308_02_T1"
            }
        ],
        "hasCustomizedMetadata": false,
        "options": {
            "bulk": true,
            "download": true,
            "order": false,
            "secondary": false
        },
        "selected": null,
        "spatialBounds": {
            "type": "Polygon",
            "coordinates": [
                [
                    [
                        -69.78196,
                        49.19588
                    ],
                    [
                        -69.78196,
                        51.34911
                    ],
                    [
                        -66.46442,
                        51.34911
                    ],
                    [
                        -66.46442,
                        49.19588
                    ],
                    [
                        -69.78196,
                        49.19588
                    ]
                ]
            ]
        },
        "spatialCoverage": {
            "type": "Polygon",
            "coordinates": [
                [
                    [
                        -69.78196,
                        49.6473
                    ],
                    [
                        -67.23413,
                        49.19588
                    ],
                    [
                        -66.46442,
                        50.89056
                    ],
                    [
                        -69.10195,
                        51.34911
                    ],
                    [
                        -69.78196,
                        49.6473
                    ]
                ]
            ]
        },
        "temporalCoverage": {
            "endDate": "2020-12-31 00:00:00",
            "startDate": "2020-12-31 00:00:00"
        },
        "publishDate": "Unknown"
    },
    "errorCode": null,
    "errorMessage": null
}
```

Scene Metadata where the input is a pre-set list.

##### Input Parameters

| Parameter Name | Data Type | Required | Description |
| --- | --- | --- | --- |
| datasetName | string | No | Used to identify the dataset to search |
| listId | string | Yes | Used to identify the list of scenes to use |
| metadataType | string | No | If populated, identifies which metadata to return (summary or full) |
| includeNullMetadataValues | boolean | No | Optional parameter to include null metadata values |
| useCustomization | boolean | No | Optional parameter to display metadata view as per user customization |
| Parameter Name | Data Type | Required | Description |

##### Examples

**Sample Request**  
```
{
    "listId": "my_comparison_list",
    "metadataType": "summary",
    "useCustomization": false
}
```

**Sample Response**  
```
{
    "data": {
        "gls_all": [
            {
                "browse": [
                    {
                        "browseName": "Standard Browse",
                        "browsePath": "https:\/\/ims.cr.usgs.gov\/browse\/gls_1990\/031\/029\/p031r029_5x19910714.jpg",
                        "overlayPath": "https:\/\/ims.cr.usgs.gov\/wms\/gls_all?sceneId=p031r029_5x19910714",
                        "overlayType": "dmid_wms",
                        "thumbnailPath": "https:\/\/ims.cr.usgs.gov\/thumbnail\/gls_1990\/031\/029\/p031r029_5x19910714.jpg",
                        "browseRotationEnabled": false
                    }
                ],
                "options": {
                    "bulk": true,
                    "order": true,
                    "download": true,
                    "secondary": false
                },
                "entityId": "P031R029_5X19910714",
                "metadata": [
                    {
                        "id": "5e7c418236ae7c32",
                        "value": "P031R029_5X19910714",
                        "fieldName": "Entity ID",
                        "dictionaryLink": null
                    }
                ],
                "selected": null,
                "displayId": "P031R029_5X19910714",
                "cloudCover": null,
                "publishDate": "2008-11-13 06:54:55-06",
                "spatialBounds": null,
                "spatialCoverage": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [
                                -101.1598325,
                                43.9842386
                            ],
                            [
                                -98.9198869,
                                43.6559118
                            ],
                            [
                                -98.3264954,
                                45.2296254
                            ],
                            [
                                -100.6272578,
                                45.5675596
                            ],
                            [
                                -101.1598325,
                                43.9842386
                            ]
                        ]
                    ]
                },
                "temporalCoverage": {
                    "endDate": "1991-07-14 00:00:00-05",
                    "startDate": "1991-07-14 00:00:00-05"
                }
            }
        ]
    },
    "version": "stable",
    "errorCode": null,
    "requestId": 27578,
    "sessionId": 1917,
    "errorMessage": ""
}
```

Returns metadata formatted in XML, ahering to FGDC, ISO and EE scene metadata formatting standards.

##### Input Parameters

| Parameter Name | Data Type | Required | Description |
| --- | --- | --- | --- |
| datasetName | string | Yes | Used to identify the dataset to search |
| entityId | string | Yes | Used to identify the scene to return results for |
| metadataType | string | No | If populated, identifies which metadata to return (full, fgdc, iso) |
| Parameter Name | Data Type | Required | Description |

##### Examples

**Sample Request**  
```
{
    "entityId": "P203R040_7X20020319",
    "datasetName": "gls_all",
    "metadataType": "fgdc"
}
```

**Sample Response**  
```
{
    "data": {
        "entityId": "P203R040_7X20020319",
        "displayId": "P203R040_7X20020319",
        "exportContent": "\n\n\t\n\t\t\n\t\t\t\n\t\t\t\tU.S. Geological Survey (USGS) Earth Resources Observation...
    },
    "version": "stable",
    "errorCode": null,
    "requestId": 28092,
    "sessionId": 1948,
    "errorMessage": ""
}
```

Searching is done with limited search criteria. All coordinates are assumed decimal-degree format. If lowerLeft or upperRight are supplied, then both must exist in the request to complete the bounding box. Starting and ending dates, if supplied, are used as a range to search data based on acquisition dates. The current implementation will only search at the date level, discarding any time information. If data in a given dataset is composite data, or data acquired over multiple days, a search will be done to match any intersection of the acquisition range. There currently is a 50,000 scene limit for the number of results that are returned, however, some client applications may encounter timeouts for large result sets for some datasets. To use the sceneFilter field, pass one of the four search filter objects (SearchFilterAnd, SearchFilterBetween, SearchFilterOr, SearchFilterValue) in JSON format with sceneFilter being the root element of the object.  
  
The response of this request includes a 'totalHits' response parameter that indicates the total number of scenes that match the search query to allow for pagination. Due to this, searches without a 'sceneFilter' parameter can take much longer to execute. To minimize this impact we use a cached scene count for 'totalHits' instead of computing the actual row count. An additional field, 'totalHitsAccuracy', is also included in the response to indicate if the 'totalHits' value was computed based off the query or using an approximated value. This does not impact the users ability to access these results via pagination. This cached value is updated daily for all datasets with active data ingests. Ingest frequency for each dataset can be found using the 'ingestFrequency' field in the dataset, dataset-categories and dataset-search endpoint responses.

##### Input Parameters

| Parameter Name | Data Type | Required | Description |
| --- | --- | --- | --- |
| datasetName | string | Yes | Used to identify the dataset to search |
| maxResults | int | No | How many results should be returned? (default = 100) |
| startingNumber | int | No | Used to identify the start number to search from |
| metadataType | string | No | If populated, identifies which metadata to return (summary or full) |
| sortField | string | No | Determines which field to sort the results on |
| sortDirection | string | No | Determines how the results should be sorted - ASC or DESC |
| sortCustomization | [SortCustomization](https://m2m.cr.usgs.gov/api/docs/datatypes/#sortCustomization) | No | Used to pass in custom sorts |
| useCustomization | boolean | No | Optional parameter to indicate whether to use customization |
| sceneFilter | [SceneFilter](https://m2m.cr.usgs.gov/api/docs/datatypes/#sceneFilter) | No | Used to filter data within the dataset |
| compareListName | string | No | If provided, defined a scene-list listId to use to track scenes selected for comparison |
| bulkListName | string | No | If provided, defined a scene-list listId to use to track scenes selected for bulk ordering |
| orderListName | string | No | If provided, defined a scene-list listId to use to track scenes selected for on-demand ordering |
| excludeListName | string | No | If provided, defined a scene-list listId to use to exclude scenes from the results |
| includeNullMetadataValues | boolean | No | Optional parameter to include null metadata values |
| Parameter Name | Data Type | Required | Description |

##### Examples

Note: It returns 100 results by default. Users can set input 'maxResults' to get different results number returned. It is recommened to set the maxResults less than 10,000 to get better performance.  
**Sample AcquisitionFilter**  
```
{
    "end": "2011-01-31",
    "start": "2010-01-01"
}
```
**Sample Requests**  
**General search**  
```
{
    "maxResults": 500,
    "datasetName": "gls_all",
    "sceneFilter": {
        "ingestFilter": null,
        "spatialFilter": null,
        "metadataFilter": null,
        "cloudCoverFilter": {
            "max": 100,
            "min": 0,
            "includeUnknown": true
        },
        "acquisitionFilter": null
    },
    "bulkListName": "my_bulk_list",
    "metadataType": "summary",
    "orderListName": "my_order_list",
    "startingNumber": 1,
    "compareListName": "my_comparison_list",
    "excludeListName": "my_exclusion_list"
}
```
**Search with spatial filter and ingest filter**  
```
{
    "maxResults": 500,
    "datasetName": "gls_all",
    "sceneFilter": {
        "spatialFilter": {
            "filterType": "mbr",
            "lowerLeft": {
                    "latitude": 40,
                    "longitude": -120
            },
            "upperRight": {
                    "latitude": 50,
                    "longitude": -100
            }
        },
        "metadataFilter": null,
        "cloudCoverFilter": {
            "max": 100,
            "min": 0,
            "includeUnknown": true
        },
        "ingestFilter": {
            "end": "2012-01-31",
            "start": "2010-01-01"
        }
    },
    "metadataType": "summary",
    "sortDirection": "ASC",
    "startingNumber": 1
}
```
**Search with acquisition filter**  
```
{
    "maxResults": 1,
    "datasetName": "gls_all",
    "sceneFilter": {
        "acquisitionFilter": {
            "end": "2011-01-31",
            "start": "2010-01-01"
        }
    },
    "metadataType": "summary",
    "sortDirection": "ASC",
    "startingNumber": 1
}
```
**Search with metadata filter (metadata filter ids can be retrieved by calling dataset-filters)**  
```
{
    "maxResults": 100,
    "datasetName": "gls_all",
    "sceneFilter": {
        "spatialFilter": null,
        "metadataFilter": {
            "filterType": "and",
            "childFilters": [
                {
                    "filterId": "5e7c418226384db5",
                    "filterType": "between",
                    "firstValue": 10,
                    "secondValue": 50
                },
                {
                    "filterId": "5e7c4182b7148b69",
                    "filterType": "between",
                    "firstValue": 50,
                    "secondValue": 90
                }
            ]
        },
        "cloudCoverFilter": {
            "max": 100,
            "min": 0,
            "includeUnknown": true
        },
        "acquisitionFilter": {
            "end": "2011-12-30",
            "start": "2011-11-30"
        }
    },
    "metadataType": "summary",
    "sortDirection": "ASC",
    "startingNumber": 1
}
```
**Sort search results using useCustomization flag and sortCustomization**  
```
{
    "maxResults": 1,
    "datasetName": "gls_all",
    "sceneFilter": {
        "acquisitionFilter": {
            "end": "2011-01-31",
            "start": "2010-01-01"
        }
    },
    "metadataType": "summary",
    "useCustomization": true,
    "sortCustomization": [
        {
            "direction": "ASC",
            "field_name": "ID"
        },
        {
            "direction": "DESC",
            "field_name": "Filepath"
        }
    ]
    "startingNumber": 1
}
```

**Sample Response**  
```
{
    "data": {
        "results": [
            {
                "browse": [
                    {
                        "browseName": "Standard Browse",
                        "browsePath": "https:\/\/ims.cr.usgs.gov\/browse\/gls_2000\/203\/042\/p203r042_7x20001210.jpg",
                        "overlayPath": "https:\/\/ims.cr.usgs.gov\/wms\/gls_all?sceneId=p203r042_7x20001210",
                        "overlayType": "dmid_wms",
                        "thumbnailPath": "https:\/\/ims.cr.usgs.gov\/thumbnail\/gls_2000\/203\/042\/p203r042_7x20001210.jpg",
                        "browseRotationEnabled": null
                    }
                ],
                "options": {
                    "bulk": true,
                    "order": true,
                    "download": true,
                    "secondary": false
                },
                "entityId": "P203R042_7X20001210",
                "metadata": [
                    {
                        "id": "5e7c418236ae7c32",
                        "value": "P203R042_7X20001210",
                        "fieldName": "Entity ID",
                        "dictionaryLink": null
                    },
                    {
                        "id": "5e7c41824cb1f9ed",
                        "value": "2000-12-10 00:00:00-06",
                        "fieldName": "Acquisition Date",
                        "dictionaryLink": null
                    },
                ],
                "selected": {
                    "bulk": false,
                    "order": false,
                    "compare": false
                },
                "displayId": "P203R042_7X20001210",
                "cloudCover": null,
                "publishDate": "2008-01-31 16:28:42-06",
                "spatialBounds": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [
                                -11.9873282,
                                25.0499806
                            ],
                            [
                                -11.9873282,
                                26.9373727
                            ],
                            [
                                -9.7765562,
                                26.9373727
                            ],
                            [
                                -9.7765562,
                                25.0499806
                            ],
                            [
                                -11.9873282,
                                25.0499806
                            ]
                        ]
                    ]
                },
                "spatialCoverage": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [
                                -11.9873282,
                                25.3127202
                            ],
                            [
                                -10.190104,
                                25.0499806
                            ],
                            [
                                -9.7765562,
                                26.6718002
                            ],
                            [
                                -11.5983405,
                                26.9373727
                            ],
                            [
                                -11.9873282,
                                25.3127202
                            ]
                        ]
                    ]
                },
                "temporalCoverage": {
                    "endDate": "2000-12-10 00:00:00-06",
                    "startDate": "2000-12-10 00:00:00-06"
                }
            },
        ],
        "totalHits": 42989,
        "totalHitsAccuracy": "approximate",
        "isCustomized": false,
        "startingNumber": 1,
        "nextRecord": 101,
        "numExcluded": 1,
        "recordsReturned": 100
    },
    "version": "stable",
    "errorCode": null,
    "requestId": 27500,
    "sessionId": 1916,
    "errorMessage": ""
}
```

This method is used to detect deleted scenes from datasets that support it. Supported datasets are determined by the 'supportDeletionSearch' parameter in the 'datasets' response. There currently is a 50,000 scene limit for the number of results that are returned, however, some client applications may encounter timeouts for large result sets for some datasets.

##### Input Parameters

| Parameter Name | Data Type | Required | Description |
| --- | --- | --- | --- |
| datasetName | string | Yes | Used to identify the dataset to search |
| maxResults | int | No | How many results should be returned? (default = 100) |
| startingNumber | int | No | Used to identify the start number to search from |
| sortField | string | No | Determines which field to sort the results on |
| sortDirection | string | No | Determines how the results should be sorted - ASC or DESC |
| temporalFilter | [TemporalFilter](https://m2m.cr.usgs.gov/api/docs/datatypes/#temporalFilter) | No | Used to filter data based on data acquisition |
| Parameter Name | Data Type | Required | Description |

##### Examples

**Sample Request**  
```
{
    "sortOrder": "ASC",
    "maxResults": 1,
    "datasetName": "landsat_8_c1",
    "startingNumber": 1,
    "temporalFilter": null
}
```

**Sample Response**  
```
{
    "data": {
        "results": [
            {
                "entityId": "LC81120292021061LGN00",
                "displayId": "LC08_L1TP_112029_20210302_20210302_01_RT",
                "deletionDate": "2021-03-11 13:00:02-06",
                "acquisitionDate": "2021-03-02 01:44:19"
            }
        ],
        "totalHits": 461,
        "nextRecord": 2,
        "recordsReturned": 1
    },
    "version": "stable",
    "errorCode": null,
    "requestId": 1233456,
    "sessionId": null,
    "errorMessage": ""
}
```

This method is used to find the related scenes for a given scene.

##### Input Parameters

| Parameter Name | Data Type | Required | Description |
| --- | --- | --- | --- |
| entityId | string | Yes | Used to identify the scene to find related scenes for |
| datasetName | string | Yes | Used to identify the dataset to search |
| maxResults | int | No | Used to identify the dataset to search |
| startingNumber | int | No | Used to identify the dataset to search |
| metadataType | string | No | If populated, identifies which metadata to return (summary or full) |
| sortField | string | No | Determines which field to sort the results on |
| sortDirection | string | No | Determines how the results should be sorted - ASC or DESC |
| compareListName | string | No | If provided, defined a scene-list listId to use to track scenes selected for comparison |
| bulkListName | string | No | If provided, defined a scene-list listId to use to track scenes selected for bulk ordering |
| orderListName | string | No | If provided, defined a scene-list listId to use to track scenes selected for on-demand ordering |
| excludeListName | string | No | If provided, defined a scene-list listId to use to exclude scenes from the results |
| Parameter Name | Data Type | Required | Description |

##### Examples

**Sample Request**  
```
{
    "entityId": "AR1GLNP10180001",
    "maxResults": "1",
    "datasetName": "aerial_combin_w",
    "startingNumber": "1"
}
```

**Sample Response**  
```
{
    "data": {
        "results": [
            {
                "browse": [
                    {
                        "id": "5e83d8f4b144e512", 
            "browseRotationEnabled": null, 
            "browseName": "Standard Browse", 
            "browsePath": "https://ims.cr.usgs.gov/browse/aircraft/phoenix/aerial/2CTQ/2CTQ03012/2CTQ03012_165.jpg", 
            "overlayPath": "https://ims.cr.usgs.gov/browse/aircraft/phoenix/aerial/2CTQ/2CTQ03012/2CTQ03012_165.jpg", 
            "overlayType": "file", 
            "thumbnailPath": "https://ims.cr.usgs.gov/thumbnail/aircraft/phoenix/aerial/2CTQ/2CTQ03012/2CTQ03012_165.jpg" 
                    }
                ],
                "options": {
                    "bulk": true, 
            "download": true, 
            "order": false, 
            "secondary": true 
                },
                "entityId": "P203R040_7X20020319",
                "displayId": "AR1GLNP1005X165", 
                "metadata": [],
                "selected": {
                    "bulk": false,
                    "order": false,
                    "compare": false
                },
                "cloudCover": 0, 
        "orderingId": null, 
                "publishDate": "2008-01-31 16:25:39-06",
                "spatialBounds": {
                    "type": "Point",
            "coordinates": [ 0, 0 ]                     
                },
                "spatialCoverage": {
                    "type": "Polygon", 
            "coordinates":  [ 
                    [ 
                        [ 0, 0 ], 
                        [ 0, 0 ], 
                        [ 0, 0 ], 
                        [ 0, 0 ], 
                        [ 0, 0 ] 
                    ] 
                    ] 
                },
                "temporalCoverage": {
                    "endDate": "1998-09-15 00:00:00-05",
                    "startDate": "1998-09-15 00:00:00-05"
                },
                "publishDate": "2017-08-28 16:17:56-05" 
            }
        ],
        "recordsReturned": 1, 
    "totalHits": 165, 
    "numExcluded": 0, 
    "startingNumber": 1, 
    "nextRecord": 2, 
    "secondaryDatasetId": "5e83d8f3f081a7ad", 
    "secondaryDatasetAlias": "aerial_combin_w" 
    },
    "version": "stable",
    "errorCode": null,
    "requestId": 28094,
    "sessionId": null,
    "errorMessage": ""
}
```

#### tram-order-detail-update

This method is used to set metadata for an order.

##### Input Parameters

| Parameter Name | Data Type | Required | Description |
| --- | --- | --- | --- |
| orderNumber | string | Yes | The order ID for the order to update |
| detailKey | string | Yes | The system detail key |
| detailValue | string | Yes | The value to store under the detailKey |
| Parameter Name | Data Type | Required | Description |

##### Examples

**Sample Request**  
```
{
    "detailKey": "my_detail_key",
    "detailValue": "test123456",
    "orderNumber": "D120pe87u1234"
}
```

**Sample Response**  
```
{
    "data": {
        "my_detail_key": "test123456"
    },
    "version": "stable",
    "errorCode": null,
    "requestId": 2355549,
    "sessionId": null,
    "errorMessage": ""
}
```

#### tram-order-details

This method is used to view the metadata within an order.

##### Input Parameters

| Parameter Name | Data Type | Required | Description |
| --- | --- | --- | --- |
| orderNumber | string | Yes | The order ID to get details for |
| Parameter Name | Data Type | Required | Description |

##### Examples

**Sample Request**  
```
{
    "orderNumber": "D120pe87u1234"
}
```

**Sample Response**  
```
{
    "data": {
        "my_detail_key_1": "test123456",
        "my_detail_key_2": "test456789"
    },
    "version": "stable",
    "errorCode": null,
    "requestId": 2357993,
    "sessionId": null,
    "errorMessage": ""
}
```

#### tram-order-details-clear

This method is used to clear all metadata within an order.

##### Input Parameters

| Parameter Name | Data Type | Required | Description |
| --- | --- | --- | --- |
| orderNumber | string | Yes | The order ID to clear details for |
| Parameter Name | Data Type | Required | Description |

##### Examples

**Sample Request**  
```
{
    "orderNumber": "D120pe87u1234"
}
```

**Sample Response**  
```
{
    "data": null,
    "version": "stable",
    "errorCode": null,
    "requestId": 2358046,
    "sessionId": null,
    "errorMessage": ""
}
```

#### tram-order-details-remove

This method is used to remove the metadata within an order.

##### Input Parameters

| Parameter Name | Data Type | Required | Description |
| --- | --- | --- | --- |
| orderNumber | string | Yes | The order ID to clear details for |
| detailKey | string | Yes | The system detail key |
| Parameter Name | Data Type | Required | Description |

##### Examples

**Sample Request**  
```
{
    "detailKey": "my_detail_key_2",
    "orderNumber": "D120pe87u1234"
}
```

**Sample Response**  
```
{
    "data": {
        "my_detail_key_1": "test123456"
    },
    "version": "stable",
    "errorCode": null,
    "requestId": 2358042,
    "sessionId": null,
    "errorMessage": ""
}
```

Search TRAM orders.

##### Input Parameters

| Parameter Name | Data Type | Required | Description |
| --- | --- | --- | --- |
| orderId | string | No | The order ID to get status for (accepts '%' wildcard) |
| maxResults | int | No | How many results should be returned on each page? (default = 25) |
| systemId | string | No | Limit results based on the application that order was submitted from |
| sortAsc | boolean | No | True for ascending results, false for descending results |
| sortField | string | No | Which field should sorting be done on? (order\_id, date\_entered or date\_updated) |
| statusFilter | string\[\] | No | An array of status codes to |
| Parameter Name | Data Type | Required | Description |

##### Examples

**Sample Request**  
```
{
    "sortAsc": false,
    "systemId": "EE",
    "sortField": "date_entered",
    "maxResults": "2"
}
```

**Sample Response**  
```
{
    "data": [
        {
            "orderId": "0121909050501",
            "username": "user",
            "statusCode": "C",
            "dateEntered": "2019-09-05T13:06:07Z",
            "orderComment": "Ordered directly from application",
            "statusCodeText": "Complete",
            "lastUpdatedDate": "2019-10-09T14:11:11Z",
            "processingPriority": 5
        },
        {
            "orderId": "0121908160492",
            "username": "user",
            "statusCode": "R",
            "dateEntered": "2019-08-16T20:19:50Z",
            "orderComment": null,
            "statusCodeText": "Rejected",
            "lastUpdatedDate": "2019-10-09T14:11:08Z",
            "processingPriority": 9
        }
    ],
    "version": "stable",
    "errorCode": null,
    "requestId": 28095,
    "sessionId": null,
    "errorMessage": ""
}
```

#### tram-order-status

Gets the status of a TRAM order.

##### Input Parameters

| Parameter Name | Data Type | Required | Description |
| --- | --- | --- | --- |
| orderNumber | string | Yes | The order ID to get status for |
| Parameter Name | Data Type | Required | Description |

##### Examples

**Sample Request**  
```
{
    "orderNumber": "P015-123456"
}
```

**Sample Response**  
```
{
    "data": {
        "units": [
            {
                "datasetId": "5e418c2a215c487e",
                "unitPrice": "30",
                "orderingId": "D3C1218-401631A002",
                "statusCode": "L",
                "unitNumber": 1,
                "datasetName": "declassiii",
                "productCode": "TS51",
                "productName": "DECLASS 3 IMAGERY ON-DEMAND",
                "unitComment": null,
                "collectionName": "Declass 3 (2013)",
                "statusCodeText": "Queued for Processing",
                "lastUpdatedDate": "2020-03-02T22:11:24Z"
            }            
        ],
        "username": "user",
        "orderNumber": "P015-123456",
        "orderStatus": "P",
        "orderStatusText": "Pending"
    },
    "version": "stable",
    "errorCode": null,
    "requestId": 4513,
    "sessionId": null,
    "errorMessage": ""
}
```

#### tram-order-units

Lists units for a specified order.

##### Input Parameters

| Parameter Name | Data Type | Required | Description |
| --- | --- | --- | --- |
| orderNumber | string | Yes | The order ID to get units for |
| Parameter Name | Data Type | Required | Description |

##### Examples

**Sample Request**  
```
{
    "orderNumber": "123456"
}
```

**Sample Response**  
```
{
    "data": [
        {
            "id": 4881376,
            "unitPrice": null,
            "orderingId": "LE71210262002064BJC00",
            "statusCode": "C",
            "statusText": "Complete",
            "unitNumber": 1,
            "datasetName": "landsat_etm_priv",
            "productCode": "LEC01",
            "productName": "ETM L1T\/L1GT COLLECTION 1 ON-DEMAND",
            "unitComment": null,
            "processingParameters": null
        }
    ],
    "version": "stable",
    "errorCode": null,
    "requestId": 26463,
    "sessionId": 1889,
    "errorMessage": ""
}
```

#### user-preference-get

This method is used to retrieve user's preference settings.

##### Input Parameters

| Parameter Name | Data Type | Required | Description |
| --- | --- | --- | --- |
| systemId | string | No | Used to identify which system to return preferences for. If null it will return all the users preferences |
| setting | string\[\] | No | If populated, identifies which setting(s) to return |
| Parameter Name | Data Type | Required | Description |

##### Examples

**Sample Request**  
```
{
    "systemId": "EE"
}
```

**Sample Response**  
```
{
    "data": "{\"map\": {\"lat\": \"43.53\", \"lng\": \"-96.73\", \"showZoom\": false, \"showMouse\": true, \"zoomLevel\": \"7\", \"defaultBasemap\": \"OpenStreetMap\"}, \"browse\": {\"browseSize\": \"10\", \"selectedOpacity\": \"100\", \"nonSelectedOpacity\": \"100\"}, \"general\": {\"defaultDataset\": \"gls_all\", \"codiscoveryEnabled\": false}}",
    "version": "stable",
    "errorCode": null,
    "requestId": 19010101,
    "sessionId": null,
    "errorMessage": ""
}
```

#### user-preference-set

This method is used to create or update user's preferences.

##### Input Parameters

| Parameter Name | Data Type | Required | Description |
| --- | --- | --- | --- |
| systemId | string | Yes | Used to identify which system the preferences are for. |
| userPreferences | string\[\] | Yes | Used to set user preferences for various systems. |
| Parameter Name | Data Type | Required | Description |

##### Examples

**Sample Request**  
```
{
    "systemId": "EE",
    "userPreferences": {
        "map": {
            "lat": "43.53",
            "lng": "-96.73",
            "showZoom": false,
            "showMouse": true,
            "zoomLevel": "7",
            "defaultBasemap": "OpenStreetMap"
        },
        "browse": {
            "browseSize": "10",
            "selectedOpacity": "100",
            "nonSelectedOpacity": "100"
        },
        "general": {
            "defaultDataset": "gls_all",
            "codiscoveryEnabled": false
        }
    }
}
```

**Sample Response**  
```
{
    "data": 1,
    "version": "experimental",
    "errorCode": null,
    "requestId": 19010007,
    "sessionId": 3437343,
    "errorMessage": ""
}
```