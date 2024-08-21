import google.generativeai as genai
from typing import Dict, List

# Manually set your API key
GOOGLE_API_KEY = 'AIzaSyBmnV48Lgg54buSbr4Pag89NmqbeFCuZ1E'
genai.configure(api_key=GOOGLE_API_KEY)

class BreastCancerDiagnosis:
    """
    A class to handle breast cancer diagnosis interpretation using Google's Generative AI.

    Attributes:
        model (genai.GenerativeModel): The Generative AI model for generating diagnostic content.
        prediction (str): The prediction result from the machine learning model.
    """
    
    def __init__(self, prediction: str) -> None:
        """
        Initializes the BreastCancerDiagnosis instance.

        Args:
            prediction (str): The prediction result from the machine learning model (e.g., 'benign' or 'malignant').
        """
        self.model = genai.GenerativeModel(model_name='gemini-1.5-flash')
        self.prediction = prediction

    def generate_diagnosis(self) -> List[str]:
        """
        Generates a detailed explanation based on the prediction result.

        Returns:
            List[str]: A list containing the detailed explanation or error messages if an exception occurs.
        """
        try:
            # Construct the prompt based on the prediction
            prompt = self._create_prompt(self.prediction)
            response = self.model.generate_content(prompt)
            
            # Ensure that response.text is returned as a list of strings
            return [response.text] if response.text else ["No explanation generated"]
        except Exception as e:
            return [f"Error generating explanation: {str(e)}"]

    def _create_prompt(self, prediction: str) -> str:
        """
        Creates a prompt for the Generative AI model based on the prediction result.

        Args:
            prediction (str): The prediction result from the machine learning model.

        Returns:
            str: The constructed prompt for generating a detailed explanation.
        """
        # Construct the prompt to explain the prediction
        prompt =f"The incidence of breast cancer is a {prediction}. Writing a message of condolence or blessing based on the situation. Note the 70 word limit."    
        return prompt

# Example usage
if __name__ == "__main__":
    # Sample prediction; replace with actual prediction
    prediction = 'Benign tumor'

    diagnosis_tool = BreastCancerDiagnosis(prediction)
    explanation = diagnosis_tool.generate_diagnosis()
    print("\n".join(explanation))
