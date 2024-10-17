from langchain_community.chat_models.fake import FakeListChatModel


class FakeListChatModelWithTools(FakeListChatModel):
  def bind_tools(self, tools):
    return self
