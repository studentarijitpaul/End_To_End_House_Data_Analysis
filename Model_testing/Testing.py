from pathlib import Path
import joblib
import pandas as pd


# Get the path of the Models folder
def get_models_folder():
    base_dir = Path(__file__).resolve().parent.parent
    return base_dir / "Models"


# Show all saved model files
def show_models(models_dir):
    model_files = sorted(models_dir.glob("*.pkl"))

    # Do not show features.pkl because it is not a model
    model_files = [file for file in model_files if file.name != "features.pkl"]

    if len(model_files) == 0:
        raise FileNotFoundError("No model files found in Models folder.")

    print("\nAvailable Models:")
    for i, file in enumerate(model_files, start=1):
        print(i, file.stem)

    return model_files


# Ask user to choose a model
def choose_model(model_files):
    while True:
        try:
            choice = int(input("\nChoose model number: "))

            if 1 <= choice <= len(model_files):
                return model_files[choice - 1]

            print("Invalid choice. Please choose a valid model number.")

        except ValueError:
            print("Please enter a number only.")


# Take integer input safely
def get_int(message):
    while True:
        try:
            return int(input(message))
        except ValueError:
            print("Please enter a valid number.")


# Take text input safely
def get_text(message):
    value = input(message).strip()

    while value == "":
        print("This field cannot be empty.")
        value = input(message).strip()

    return value


# Take house details from user
def take_user_input():
    data = {
        "State": get_text("State: "),
        "City": get_text("City: "),
        "Locality": get_text("Locality: "),
        "Property_Type": get_text("Property Type: "),
        "BHK": get_int("BHK: "),
        "Size_in_SqFt": get_int("Size in SqFt: "),
        "Year_Built": get_int("Year Built: "),
        "Furnished_Status": get_text("Furnished Status: "),
        "Floor_No": get_int("Floor No: "),
        "Total_Floors": get_int("Total Floors: "),
        "Age_of_Property": get_int("Age of Property: "),
        "Nearby_Schools": get_int("Nearby Schools: "),
        "Nearby_Hospitals": get_int("Nearby Hospitals: "),
        "Public_Transport_Accessibility": get_text("Public Transport Accessibility: "),
        "Parking_Space": get_text("Parking Space: "),
        "Security": get_text("Security: "),
        "Amenities": get_text("Amenities: "),
        "Facing": get_text("Facing: "),
        "Owner_Type": get_text("Owner Type: "),
        "Availability_Status": get_text("Availability Status: ")
    }

    return pd.DataFrame([data])


# Convert text columns and match training columns
def prepare_input(input_df, columns):
    input_df = pd.get_dummies(input_df)
    input_df = input_df.reindex(columns=columns, fill_value=0)
    return input_df


# Main program
def main():
    try:
        # Get Models folder
        models_dir = get_models_folder()

        # Load feature columns used during training
        columns = joblib.load(models_dir / "features.pkl")

        # Show models and let user choose one
        model_files = show_models(models_dir)
        model_path = choose_model(model_files)

        # Load selected model
        model = joblib.load(model_path)

        # Take user input
        user_df = take_user_input()

        # Prepare input for model
        final_input = prepare_input(user_df, columns)

        # Predict price
        prediction = model.predict(final_input)

        # Calculate estimated price per square foot
        price_per_sqft = (prediction[0] * 100000) / user_df["Size_in_SqFt"][0]

        # Show result
        print("\nPrediction Result")
        print("Model Used:", model_path.stem)
        print("Predicted Price in Lakhs:", round(prediction[0], 2))
        print("Estimated Price per SqFt:", round(price_per_sqft, 2))

    except FileNotFoundError as error:
        print("File error:", error)

    except Exception as error:
        print("Something went wrong:", error)


# Start program
main()