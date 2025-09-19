from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union
from google.adk.agents import LlmAgent
from google.adk.tools import FunctionTool
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext
import structlog
from config.settings import settings

logger = structlog.get_logger(__name__)

class BaseUseCaseAgent(ABC):
    """
    Abstract base class for all use case agents.
    
    Why this design:
    1. Enforces consistent structure across all use cases
    2. Provides common functionality (context injection, logging)
    3. Enables polymorphic usage in experiments
    4. Follows ADK patterns established in your codebase
    """
    
    def __init__(self, model: str, use_case: str, role: str):
        self.model = model
        self.use_case = use_case  # e.g., "system_design"
        self.role = role          # e.g., "producer", "critic"
        self.name = f"{use_case}_{role}"
        
        # Initialize components
        self.tools = self._initialize_tools()
        self.agent = self._create_adk_agent()
        
        logger.info(
            "Agent initialized",
            agent_name=self.name,
            model=self.model,
            use_case=self.use_case,
            role=self.role,
            tool_count=len(self.tools)
        )
    
    @abstractmethod
    def _initialize_tools(self) -> List[BaseTool]:
        """
        Initialize tools specific to this agent's role and use case.
        
        Why abstract: Each agent needs different tools based on its purpose.
        Examples:
        - System design producer: pricing tools, architecture tools
        - System design critic: security analysis, best practices checker
        - Content producer: research tools, style guides
        - Content critic: grammar checker, fact checker
        """
        pass
    
    @abstractmethod
    def _get_instructions(self) -> str:
        """
        Get agent-specific instructions.
        
        Why abstract: Instructions must be tailored to:
        1. Use case domain knowledge
        2. Agent role (producer vs critic)
        3. Expected output format
        4. Quality standards
        """
        pass
    
    def _create_adk_agent(self) -> LlmAgent:
        """
        Create the ADK LlmAgent instance following established patterns.
        
        Why this pattern:
        1. Consistent with your existing codebase
        2. Proper ADK integration
        3. Common callback handling
        4. Error handling and logging
        """
        try:
            agent = LlmAgent(
                model=self.model,
                name=self.name,
                description=f"{self.use_case} {self.role} agent",
                instruction=self._get_instructions(),
                tools=self.tools,
                before_tool_callback=self._before_tool_callback
            )
            
            logger.info(
                "ADK agent created successfully",
                agent_name=self.name,
                model=self.model
            )
            return agent
            
        except Exception as e:
            logger.error(
                "Failed to create ADK agent",
                agent_name=self.name,
                model=self.model,
                error=str(e)
            )
            raise
    
    def _before_tool_callback(self, tool: BaseTool, args: Dict[str, Any], 
                             tool_context: ToolContext) -> None:
        """
        Inject common context into all tool calls.
        
        Why this pattern (from your codebase):
        1. Consistent context propagation
        2. Research experiment tracking
        3. Cost and usage monitoring
        4. Authentication and session management
        """
        # Research experiment tracking
        experiment_id = tool_context.state.get("experiment:id")
        if experiment_id:
            args["experiment_id"] = experiment_id
        
        # Iteration context for reflection research
        iteration_count = tool_context.state.get("reflection:iteration", 0)
        args["iteration_context"] = iteration_count
        
        # Cost tracking for research
        if settings.cost_tracking_enabled:
            args["track_usage"] = True
            args["use_case"] = self.use_case
            args["agent_role"] = self.role
            args["model"] = self.model
        
        # Session and auth context (similar to your codebase)
        session_id = tool_context.state.get("session:session_id")
        if session_id:
            args["session_id"] = session_id
        
        # User context for personalization
        user_context = tool_context.state.get("user:context", {})
        if user_context:
            args["user_context"] = user_context
        
        logger.debug(
            "Tool context injected",
            tool_name=tool.name,
            agent_name=self.name,
            experiment_id=experiment_id,
            iteration=iteration_count
        )
    
    async def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the agent with input data using proper ADK patterns.
        
        Why this wrapper:
        1. Consistent interface across all agents
        2. Error handling and logging
        3. Performance monitoring
        4. Result standardization
        """
        try:
            logger.info(
                "Agent execution started",
                agent_name=self.name,
                input_keys=list(input_data.keys())
            )
            
            # Import proper ADK components
            from google.adk.runners import Runner
            from google.adk.sessions import DatabaseSessionService
            from google.genai import types
            from config.settings import settings
            
            # Create session service (using DatabaseSessionService as recommended)
            session_service = DatabaseSessionService(db_url=settings.database_url or "sqlite:///research.db")
            
            # Create session
            user_id = "research_user"
            session = await session_service.create_session(
                app_name=self.name,
                user_id=user_id
            )
            
            # Create runner with proper app_name matching
            runner = Runner(
                app_name=self.name,
                agent=self.agent,
                session_service=session_service
            )
            
            # Create proper ADK Content object
            message_text = input_data.get("input", str(input_data))
            content = types.Content(role="user", parts=[types.Part(text=message_text)])
            
            # Execute agent
            invocation_events = []
            async for evt in runner.run_async(
                user_id=session.user_id,
                session_id=session.id,
                new_message=content
            ):
                invocation_events.append(evt)
            
            # Extract final response using proper event handling
            final_response = self._extract_text_from_events(invocation_events)
            
            result = {
                "response": final_response,
                "session_id": session.id,
                "events_count": len(invocation_events),
                "agent_name": self.name
            }
            
            logger.info(
                "Agent execution completed",
                agent_name=self.name,
                response_length=len(final_response) if final_response else 0,
                events_count=len(invocation_events)
            )
            
            return result
            
        except Exception as e:
            logger.error(
                "Agent execution failed",
                agent_name=self.name,
                error=str(e),
                input_data=input_data
            )
            raise
    
    def _extract_text_from_events(self, events: List[Any]) -> Optional[str]:
        """
        Extract text content from ADK events, handling function calls properly.
        Based on working patterns from expert guidance.
        """
        def extract_text_from_event(evt) -> Optional[str]:
            content = getattr(evt, "content", None)
            if content is None:
                return None

            parts = getattr(content, "parts", None)
            if not parts:
                return None

            first_part = parts[0]
            if getattr(first_part, "function_call", None) is not None:
                return None  # Skip function calls

            return getattr(first_part, "text", None)
        
        # Extract final response from events
        final_response = None
        for evt in events:
            text = extract_text_from_event(evt)
            if text:
                final_response = text
        
        return final_response
    
    def update_model(self, new_model: str) -> None:
        """
        Update the model for this agent.
        
        Why needed: Research experiments test different models
        on the same agent configuration.
        """
        if new_model not in settings.available_models:
            raise ValueError(f"Model {new_model} not in available models: {settings.available_models}")
        
        old_model = self.model
        self.model = new_model
        
        # Recreate ADK agent with new model
        self.agent = self._create_adk_agent()
        
        logger.info(
            "Agent model updated",
            agent_name=self.name,
            old_model=old_model,
            new_model=new_model
        )

