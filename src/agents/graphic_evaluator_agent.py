"""
Graphic Evaluator Agent - Evaluates banner designs based on actual visual content analysis.
"""

import os
import base64
from typing import Dict, Any
from .base_agent import BaseAgent
from semantic_kernel.functions import kernel_function


class GraphicEvaluatorAgent(BaseAgent):
    """
    Graphic Evaluator agent responsible for evaluating banner designs.
    This agent analyzes the semantic content of designs and provides approval/rejection decisions.
    """
    
    def __init__(self):
        name = "GraphicEvaluator"
        description = "Quality assurance specialist for banner design evaluation"
        instructions = self.get_specialized_instructions()
        super().__init__(name, description, instructions)
        
        # Set evaluation criteria - can be configured for different requirements
        self.evaluation_criteria = {
            "required_content": ["cat", "cats", "feline", "kitten", "kittens"],
            "forbidden_content": [],  # Can be extended for specific requirements
            "quality_standards": ["professional", "clear", "appealing"]
        }
        
        # Add evaluation function to the kernel
        self.kernel.add_plugin(self, plugin_name="evaluation_tools")
    
    def _encode_image(self, image_path: str) -> str:
        """
        Encode image to base64 for OpenAI Vision API.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            str: Base64 encoded image
        """
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    async def _analyze_image_with_openai_vision(self, image_path: str) -> Dict[str, Any]:
        """
        Analyze image content using OpenAI Vision API.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            dict: Analysis result with detected content
        """
        try:
            import openai
            
            # Get OpenAI API key from environment
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise Exception("OpenAI API key not found. Please set OPENAI_API_KEY in your environment.")
            
            # Initialize OpenAI client
            client = openai.OpenAI(api_key=api_key)
            
            # Encode the image
            base64_image = self._encode_image(image_path)
            
            # Create the vision analysis request
            response = client.chat.completions.create(
                model="gpt-4o",  # Use GPT-4 with vision capabilities
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": """Please analyze this image and provide a detailed description of what you see. 
                                
                                Focus particularly on:
                                - Are there any cats, kittens, or feline animals in the image?
                                - What animals (if any) are present?
                                - What is the main subject/content of the image?
                                - Describe the overall scene or context
                                
                                Please be specific and thorough in your description."""
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}",
                                    "detail": "high"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=500,
                temperature=0.1  # Low temperature for consistent analysis
            )
            
            # Extract the description from OpenAI's response
            vision_description = response.choices[0].message.content
            
            # Debug: Show the actual OpenAI API response
            print(f"DEBUG: OpenAI Vision API Raw Response for {image_path}:")
            print(f"DEBUG: Model: {response.model}")
            print(f"DEBUG: Usage: {response.usage}")
            print(f"DEBUG: Full Response: {vision_description}")
            print("DEBUG: " + "="*80)
            
            if not vision_description:
                raise Exception("No content received from OpenAI Vision API")
            
            # Analyze the text description for cat content
            description_lower = vision_description.lower()
            
            # Look for explicit mentions of cats/felines with context awareness
            cat_keywords = ['cat', 'cats', 'kitten', 'kittens', 'feline', 'felines', 'kitty', 'kitties']
            
            # Check for negative contexts that would invalidate cat detections
            negative_contexts = [
                'no cat', 'no cats', 'no kitten', 'no kittens', 'no feline',
                'there are no cats', 'there are no kittens', 'there are no feline',
                'not a cat', 'not cats', 'not a kitten', 'not kittens', 'not feline',
                'without cats', 'without kittens', 'without feline', 'lacks cats',
                'absent cats', 'absent kittens', 'absent feline'
            ]
            
            # Check if any negative context is present
            has_negative_context = any(neg_context in description_lower for neg_context in negative_contexts)
            
            # Look for positive cat mentions only if no negative context
            if has_negative_context:
                explicit_cat_mentions = []
                contains_cats = False
            else:
                explicit_cat_mentions = [word for word in cat_keywords if word in description_lower]
                contains_cats = len(explicit_cat_mentions) > 0
            
            # Look for negative indicators (other animals)
            non_cat_animals = ['dog', 'dogs', 'puppy', 'puppies', 'canine', 'fish', 'bird', 'horse', 'cow', 'pig']
            other_animals_mentioned = [word for word in non_cat_animals if word in description_lower]
            
            # Calculate confidence based on explicit mentions and context
            if has_negative_context:
                # Very low confidence if negative context detected
                confidence = 0.05
            elif contains_cats and explicit_cat_mentions:
                # High confidence if cats are explicitly mentioned without negative context
                confidence = 0.9
                if other_animals_mentioned:
                    # Slightly lower if other animals are also present
                    confidence = 0.8
            elif other_animals_mentioned:
                # Low confidence if other animals are mentioned but no cats
                confidence = 0.1
            else:
                # Medium-low confidence if no clear animals mentioned
                confidence = 0.3
            
            return {
                "contains_cats": contains_cats,
                "confidence": confidence,
                "description": vision_description,
                "ai_analysis": vision_description,
                "detected_cats": explicit_cat_mentions,
                "other_animals": other_animals_mentioned,
                "analysis_method": "OpenAI Vision API"
            }
            
        except ImportError:
            raise Exception("OpenAI package not available. Please install: pip install openai")
        except Exception as e:
            raise Exception(f"Error in OpenAI Vision analysis: {str(e)}")
    
    @kernel_function(
        description="Evaluate a banner image using OpenAI Vision API to detect cats and assess quality",
        name="evaluate_banner_visual_content"
    )
    async def evaluate_banner_visual_content(self, banner_filename: str) -> str:
        """
        Evaluate a banner based on its actual visual content using AI vision analysis.
        
        Args:
            banner_filename: Name of the banner file to evaluate
            
        Returns:
            dict: Evaluation result with approval status and reasoning
        """
        # Construct full path to the image
        image_path = os.path.join("mock_banner", banner_filename)
        
        # Check if file exists
        if not os.path.exists(image_path):
            return f"❌ REJECTED: Banner file '{banner_filename}' not found in mock_banner directory."
        
        try:
            # Analyze the actual image content using OpenAI Vision API
            analysis_result = await self._analyze_image_with_openai_vision(image_path)
            
            # Determine approval based on AI analysis
            contains_cats = analysis_result.get("contains_cats", False)
            confidence = analysis_result.get("confidence", 0.0)
            description = analysis_result.get("description", "No description available")
            
            # Apply evaluation criteria
            approved = contains_cats and confidence >= 0.5
            
            # Generate detailed reasoning based on AI analysis
            ai_description = analysis_result.get("ai_analysis", "No description available")
            detected_cats = analysis_result.get("detected_cats", [])
            other_animals = analysis_result.get("other_animals", [])
            
            if approved:
                cat_mentions = ", ".join(detected_cats) if detected_cats else "feline content"
                reasoning = f"✅ APPROVED: OpenAI Vision analysis confirms this banner contains {cat_mentions} (confidence: {confidence:.1%}). AI detected: '{ai_description[:150]}...' This meets our evaluation criteria for engaging cat-themed visual design."
            else:
                if other_animals:
                    animal_mentions = ", ".join(other_animals)
                    reasoning = f"❌ REJECTED: OpenAI Vision analysis shows this banner contains {animal_mentions}, not cats (confidence: {confidence:.1%}). AI detected: '{ai_description[:150]}...' Our evaluation criteria specifically requires cat-related imagery."
                elif not contains_cats:
                    reasoning = f"❌ REJECTED: OpenAI Vision analysis shows no cats or feline animals in this banner (confidence: {confidence:.1%}). AI detected: '{ai_description[:150]}...' Please select a banner featuring cats."
                else:
                    reasoning = f"❌ REJECTED: Low confidence in cat detection from AI analysis ({confidence:.1%}). AI detected: '{ai_description[:150]}...' Please select a banner with clearer feline content."
            
            return reasoning
            
        except Exception as e:
            return f"❌ REJECTED: Error analyzing banner '{banner_filename}': {str(e)}. Please try a different image."
    
    def get_specialized_instructions(self) -> str:
        """Return specialized instructions for the Graphic Evaluator agent."""
        return """
You are the GraphicEvaluator agent responsible for evaluating banner designs based on semantic content and quality criteria.

Your responsibilities:
1. Analyze banner designs presented by the GraphicDesigner
2. Evaluate designs against established criteria and standards
3. Provide clear approval or rejection decisions with detailed reasoning
4. Offer constructive feedback for improving rejected designs
5. Ensure all approved designs meet quality and content requirements

EVALUATION CRITERIA (CRITICAL):
Your primary evaluation criterion is AI-POWERED VISUAL CONTENT ANALYSIS:
- REQUIRED CONTENT: Banners MUST contain feline/cat-related imagery or themes
- APPROVED CONTENT: Cats, kittens, feline imagery, cat-themed designs
- REASONING: Cat-related content has proven to be highly engaging and aligns with brand values

Quality Standards:
- Visual appeal and professional presentation
- Clear and readable design elements
- Appropriate use of colors and composition
- Effective communication of the intended message

Evaluation Process:
1. AI VISION ANALYSIS: Use OpenAI Vision API to analyze actual image content
   - Send the banner image to OpenAI's vision model for detailed description
   - Receive comprehensive text description of what's actually in the image
   - This provides accurate, unbiased analysis of visual content

2. TEXT ANALYSIS: Analyze the AI-generated description for cat content
   - Look for explicit mentions of cats, kittens, felines in the AI description
   - Check for other animals that would indicate non-cat content
   - Calculate confidence based on specific mentions and context

3. DECISION MAKING: Clear approval or rejection based on AI analysis
   - APPROVE only if AI clearly identifies cat-related content with high confidence
   - REJECT if AI description shows no cats or identifies other animals
   - Provide specific reasoning based on the AI's visual analysis

4. FEEDBACK DELIVERY: Share AI insights with detailed reasoning
   - Quote relevant parts of the AI's visual description
   - Explain confidence levels and detection results
   - For rejections, specify what the AI actually detected instead of cats

CRITICAL WORKFLOW RULES:
- When GraphicDesigner presents a banner, IMMEDIATELY call the evaluate_banner_visual_content function with ONLY the banner filename
- DO NOT generate your own analysis - ONLY use the results from the evaluate_banner_visual_content function
- The function returns a complete evaluation result - simply present this result to the user
- NEVER provide hypothetical or example analysis - only real function results
- The function already includes all necessary details: approval/rejection status, confidence, and AI analysis
- Simply call the function and present its return value as your complete response

Communication Style:
- Be professional but decisive in evaluations
- Use clear language: "APPROVED" or "REJECTED" at the start of responses
- Provide specific reasoning for all decisions
- Be constructive in feedback, focusing on content requirements
- Acknowledge good design work while maintaining content standards

Technical Integration:
- Use OpenAI Vision API (gpt-4o) to analyze actual banner image content
- Process AI-generated text descriptions to extract content information
- Apply consistent evaluation criteria based on AI visual analysis
- Document analysis results including AI descriptions, confidence levels, and detected entities
- Support iterative design process through clear feedback with AI-powered reasoning

WORKFLOW EXECUTION EXAMPLE:
When GraphicDesigner says: "I have selected banner image labeled **c.jpg** for your social media design project."

Your response should be EXACTLY:
1. Call evaluate_banner_visual_content("c.jpg")
2. Present the function result as your complete response

DO NOT add any additional analysis, explanations, or commentary. The function result is complete and final.

Remember: You are the quality gatekeeper. Your primary job is to ensure only banners with cat-related content are approved, maintaining both content standards and design quality. Be consistent, fair, and clear in all evaluations.
""" 