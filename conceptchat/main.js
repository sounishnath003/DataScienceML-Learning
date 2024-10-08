import { sendMessage } from "./api";
import "./style.css";

const chatHistory = [
  { message: "Hey How are you today?", agent: "LLM" },
  { message: "Hope you are doing well sir!!.", agent: "LLM" },
];
const app = document.querySelector("#app");

app.innerHTML = `
    <div class="flex h-screen antialiased text-gray-900">
      <div
        class="flex flex-row h-full w-full overflow-x-hidden overflow-y-scroll"
      >
        <div class="flex flex-col py-8 pl-6 pr-2 w-64 bg-white flex-shrink-0">
          <div class="flex flex-row items-center justify-center h-12 w-full">
            <div
              class="flex items-center justify-center rounded-2xl text-blue-700 bg-blue-100 h-10 w-10"
            >
              <svg
                class="w-6 h-6"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
                xmlns="http://www.w3.org/2000/svg"
              >
                <path
                  stroke-linecap="round"
                  stroke-linejoin="round"
                  stroke-width="2"
                  d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z"
                ></path>
              </svg>
            </div>
            <div class="ml-2 font-bold text-2xl">DocsLLM</div>
          </div>
          <div
            class="flex flex-col items-center bg-blue-100 border border-gray-200 mt-4 w-full py-6 px-4 rounded-lg"
          >
            <div class="h-20 w-20 rounded-full border overflow-hidden">
              <img
                src="https://avatars3.githubusercontent.com/u/2763884?s=128"
                alt="Avatar"
                class="h-full w-full"
              />
            </div>
            <div class="text-sm font-semibold mt-2">Aminos Co.</div>
            <div class="text-xs text-gray-500">Lead UI/UX Designer</div>
            <div class="flex flex-row items-center mt-3">
              <div
                class="flex flex-col justify-center h-4 w-8 bg-blue-600 rounded-full"
              >
                <div class="h-3 w-3 bg-white rounded-full self-end mr-1"></div>
              </div>
              <div class="leading-none ml-1 text-xs">Active</div>
            </div>
          </div>
          <div class="flex flex-col mt-8">
            <div class="flex flex-row items-center justify-between text-xs">
              <span class="font-bold">Active Workspaces</span>
              <span
                class="flex items-center justify-center bg-gray-300 h-4 w-4 rounded-full"
                >&plus;</span
              >
            </div>
            <div
              class="flex flex-col space-y-1 mt-4 -mx-2 h-48 overflow-y-auto"
            >
              <button
                class="flex flex-row items-center hover:bg-gray-100 rounded-xl p-2"
              >
                <div
                  class="flex items-center justify-center h-8 w-8 bg-blue-200 rounded-full"
                >
                  D
                </div>
                <div class="ml-2 text-sm font-semibold">Database licensing</div>
              </button>
            </div>
          </div>
        </div>
        <div class="flex flex-col flex-auto h-full p-6">
          <div
            class="flex flex-col flex-auto flex-shrink-0 rounded-2xl bg-gray-100 h-full p-4"
          >
            <div class="flex flex-col h-full overflow-x-auto mb-4" id="chatContainer">
              <div class="flex flex-col h-full">
              <div id="messagesContainer">
                <div class="grid grid-cols-12 gap-y-2">
                    <div class="col-start-1 col-end-8 p-3 rounded-lg">
                      <div class="flex flex-row items-center">
                        <div
                          class="flex items-center justify-center h-10 w-10 rounded-full bg-gray-200 flex-shrink-0"
                        >
                          A
                        </div>
                        <div
                          class="relative ml-3 text-sm bg-white py-2 px-4 shadow rounded-xl"
                        >
                          <div>Hey How are you today?</div>
                        </div>
                      </div>
                    </div>
                    <div class="col-start-1 col-end-8 p-3 rounded-lg">
                      <div class="flex flex-row items-center">
                        <div
                          class="flex items-center justify-center h-10 w-10 rounded-full bg-gray-200 flex-shrink-0"
                        >
                          A
                        </div>
                        <div
                          class="relative ml-3 text-sm bg-white py-2 px-4 shadow rounded-xl"
                        >
                          <div>
                            I am an intellegent advisory agent from Internal management. Especially designed
                            and optimized for your project. You can ask me anything.
                          </div>
                        </div>
                      </div>
                    </div>
                    <div class="col-start-6 col-end-13 p-3 rounded-lg">
                      <div
                        class="flex items-center justify-start flex-row-reverse"
                      >
                        <div
                          class="flex items-center justify-center h-10 w-10 rounded-full bg-blue-600 flex-shrink-0"
                        >
                          Y
                        </div>
                        <div
                          class="relative mr-3 text-sm bg-blue-100 py-2 px-4 shadow rounded-xl"
                        >
                          <div>I'm ok what about you?</div>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
            <div
              class="flex flex-row items-center h-16 rounded-xl bg-white w-full px-4"
            >
              <div class="flex-grow ml-4">
                <div class="relative w-full">
                  <input
                    type="text"
                    id="chatmessage"
                    class="flex w-full border rounded-xl focus:outline-none focus:border-blue-300 pl-4 h-10"
                    placeholder="Ask me anything..."
                  />
                  <button
                    class="absolute flex items-center justify-center h-full w-12 right-0 top-0 text-gray-400 hover:text-gray-600"
                  >
                    <svg
                      class="w-6 h-6"
                      fill="none"
                      stroke="currentColor"
                      viewBox="0 0 24 24"
                      xmlns="http://www.w3.org/2000/svg"
                    >
                      <path
                        stroke-linecap="round"
                        stroke-linejoin="round"
                        stroke-width="2"
                        d="M14.828 14.828a4 4 0 01-5.656 0M9 10h.01M15 10h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
                      ></path>
                    </svg>
                  </button>
                </div>
              </div>
              <div class="ml-4">
                <button
                  id="submitbutton"
                  class="flex items-center justify-center bg-blue-600 hover:bg-blue-600 rounded-xl text-white px-4 py-1 flex-shrink-0"
                >
                  <span>Send</span>
                  <span class="ml-2">
                    <svg
                      class="w-4 h-4 transform rotate-45 -mt-px"
                      fill="none"
                      stroke="currentColor"
                      viewBox="0 0 24 24"
                      xmlns="http://www.w3.org/2000/svg"
                    >
                      <path
                        stroke-linecap="round"
                        stroke-linejoin="round"
                        stroke-width="2"
                        d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8"
                      ></path>
                    </svg>
                  </span>
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
`;

app
  .querySelector("#chatmessage")
  .addEventListener("keypress", function (event) {
    if (event.key === "Enter") {
      event.preventDefault();
      event.stopPropagation();
      app.querySelector("#submitbutton").click();
      event.target.value = "";
    }
  });

app.querySelector("#submitbutton").addEventListener("click", function (event) {
  event.preventDefault();
  event.stopPropagation();
  let message = document.querySelector("#chatmessage").value;
  if (message === null || message.length === 0) return;

  chatHistory.push({ message, agent: "Human" });
  let messagesContainer = app.querySelector("#messagesContainer");
  console.log(messagesContainer);
  let humanElem = document.createElement("div");
  humanElem.innerHTML = `
    <div class="col-start-6 col-end-13 p-3 rounded-lg">
      <div
        class="flex items-center justify-start flex-row-reverse"
      >
        <div
          class="flex items-center justify-center h-10 w-10 rounded-full bg-blue-600 flex-shrink-0"
        >
          Y
        </div>
        <div
          class="relative mr-3 text-sm bg-blue-100 py-2 px-4 shadow rounded-xl"
        >
          <div>${message}</div>
          <div class="absolute text-xs bottom-0 right-0 -mb-5 mr-2 text-gray-500">Seen</div>
        </div>
      </div>
    </div>
  `;
  messagesContainer.appendChild(humanElem);
  const response = sendMessage(message);
  chatHistory.push({ message: response, agent: "LLM" });

  let agentElem = document.createElement("div");
  agentElem.innerHTML = `
    <div class="col-start-1 col-end-8 p-3 rounded-lg">
      <div class="flex flex-row items-center">
        <div
          class="flex items-center justify-center h-10 w-10 rounded-full bg-gray-200 flex-shrink-0"
        >
          A
        </div>
        <div
          class="relative ml-3 text-sm bg-white py-2 px-4 shadow rounded-xl"
        >
          <div>${response}</div>
        </div>
      </div>
    </div>
  `;
  messagesContainer.appendChild(agentElem);

  app.querySelector("#chatContainer").scrollTop =
    app.querySelector("#chatContainer").scrollHeight;

  console.log({ chatHistory });

  app.querySelector("chatmessage").value = "";
});
