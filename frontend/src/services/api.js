// API Service for interacting with FastAPI backend
const API_BASE_URL = "http://localhost:8000";

/**
 * Uploads a file to the backend
 * @param {File} file 
 * @returns {Promise<Object>}
 */
export const uploadFile = async (file) => {
    const formData = new FormData();
    formData.append("file", file);

    const response = await fetch(`${API_BASE_URL}/upload`, {
        method: "POST",
        body: formData,
    });

    if (!response.ok) {
        throw new Error("Failed to upload file");
    }

    return response.json();
};

/**
 * Requests predictions/metrics for a given dataset reference
 * @param {string} datasetRef 
 * @returns {Promise<Object>}
 */
export const getPredictions = async (datasetRef) => {
    const response = await fetch(`${API_BASE_URL}/predict`, {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify({ dataset_ref: datasetRef }),
    });

    if (!response.ok) {
        throw new Error("Failed to get predictions");
    }

    return response.json();
};

/**
 * Fetches insights for a given dataset reference
 * @param {string} datasetRef 
 * @returns {Promise<Object>}
 */
export const getInsights = async (datasetRef) => {
    const response = await fetch(`${API_BASE_URL}/insights?dataset_ref=${datasetRef}`);

    if (!response.ok) {
        throw new Error("Failed to get insights");
    }

    return response.json();
};
