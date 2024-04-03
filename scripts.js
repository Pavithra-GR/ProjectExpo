function calculateCarbonFootprint() {
    const distance = document.getElementById('distance').value;
    const vehicle = document.getElementById('vehicle').value;
    let emissionFactor;
  
    switch (vehicle) {
      case 'car':
        emissionFactor = 0.3; // kg CO2 per km
        break;
      case 'motorcycle':
        emissionFactor = 0.2; // kg CO2 per km
        break;
      case 'bus':
        emissionFactor = 0.3; // kg CO2 per km
        break;
      case 'Train':
          emissionFactor = 0.4; // kg CO2 per km
          break;
      case 'bicycle':
        emissionFactor = 0; // kg CO2 per km
        break;
      default:
        emissionFactor = 0;
    }
  
    const carbonFootprint = distance * emissionFactor;
    document.getElementById('result').textContent = carbonFootprint.toFixed(2);
  }
  